"""Integration-ish tests for ``read_parquet()`` on the DataSourceV2 path.

These tests exercise planning-time behavior: schema inference,
``ListFiles → ReadFiles`` attachment to the logical plan, and
unsupported-option gating. They call ``ray.data.read_parquet`` which
triggers Ray auto-init, so they live alongside the other datasource
integration tests rather than under ``tests/unit/``.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
from ray.data import FileShuffleConfig
from ray.data._internal.datasource_v2.scanners.parquet_scanner import ParquetScanner
from ray.data._internal.logical.operators import ListFiles, ReadFiles
from ray.data.context import DataContext


def _write(path, table):
    pq.write_table(table, str(path))


@pytest.fixture
def restore_ctx():
    ctx = DataContext.get_current()
    original_use_datasource_v2 = ctx.use_datasource_v2
    original_read_op_min_num_blocks = ctx.read_op_min_num_blocks
    original_cap_enabled = ctx.read_files_estimated_num_outputs_cap_enabled
    try:
        yield ctx
    finally:
        ctx.use_datasource_v2 = original_use_datasource_v2
        ctx.read_op_min_num_blocks = original_read_op_min_num_blocks
        ctx.read_files_estimated_num_outputs_cap_enabled = original_cap_enabled


def test_v2_flag_default():
    # The default is driven by ``DEFAULT_USE_DATASOURCE_V2``. Asserting
    # either direction here would be brittle, so just check that the
    # default is a bool.
    ctx = DataContext()
    assert isinstance(ctx.use_datasource_v2, bool)


def test_read_parquet_v2_count_from_manifest(tmp_path, restore_ctx):
    # count() on a bare V2 parquet read is answered from the listing manifest
    # (footer-derived row counts) without materializing data, and equals the
    # real row count across multiple files.
    _write(tmp_path / "a.parquet", pa.table({"a": list(range(10))}))
    _write(tmp_path / "b.parquet", pa.table({"a": list(range(25))}))

    restore_ctx.use_datasource_v2 = True
    ds = ray.data.read_parquet(str(tmp_path))

    # The fast path fires and returns the exact count.
    assert ds._try_count_from_manifest() == 35
    # End-to-end count() agrees.
    assert ds.count() == 35


def test_read_parquet_v2_count_falls_back_with_downstream_op(tmp_path, restore_ctx):
    # A row-changing operator above the read (filter / limit) means the plan is
    # no longer a bare ReadFiles, so the fast path declines and count() returns
    # the true post-op count via the slow path.
    from ray.data.expressions import col

    _write(tmp_path / "a.parquet", pa.table({"a": list(range(10))}))

    restore_ctx.use_datasource_v2 = True
    filtered = ray.data.read_parquet(str(tmp_path)).filter(expr=col("a") >= 7)
    assert filtered._try_count_from_manifest() is None
    assert filtered.count() == 3

    limited = ray.data.read_parquet(str(tmp_path)).limit(4)
    assert limited._try_count_from_manifest() is None
    assert limited.count() == 4


def test_read_parquet_builds_list_files_read_files_chain(tmp_path, restore_ctx):
    f = tmp_path / "data.parquet"
    _write(f, pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}))

    restore_ctx.use_datasource_v2 = True
    ds = ray.data.read_parquet(str(tmp_path))

    assert isinstance(ds._logical_plan.dag, ReadFiles)
    assert isinstance(ds._logical_plan.dag.input_dependencies[0], ListFiles)
    schema = ds.schema()
    assert schema is not None
    assert "a" in schema.names
    assert "b" in schema.names


def test_read_parquet_v2_infer_metadata_size_bytes_is_populated(tmp_path, restore_ctx):
    # Regression test: a bare V2 ReadFiles used to always report
    # infer_metadata().size_bytes=None, which forces hash-shuffle/join
    # aggregator memory sizing onto an online sample that can severely
    # under-estimate early in execution (confirmed via a live A/B run: V2
    # underestimated a dataset's size by ~9x relative to V1 for the same
    # query, plausibly causing a long tail). A real, non-None (if
    # approximate) size estimate should be available at plan time.
    for i in range(5):
        _write(tmp_path / f"f{i}.parquet", pa.table({"a": list(range(1000))}))

    restore_ctx.use_datasource_v2 = True
    ds = ray.data.read_parquet(str(tmp_path))

    meta = ds._logical_plan.dag.infer_metadata()
    assert meta.size_bytes is not None
    assert meta.size_bytes > 0

    # Dataset.size_bytes() is a downstream consumer of this same signal.
    assert ds.size_bytes() is not None
    assert ds.size_bytes() > 0


def test_read_parquet_v2_infer_metadata_size_bytes_propagates_through_project(
    tmp_path, restore_ctx
):
    # The fix's actual target: HashAggregate/Join sit behind a `Project` in
    # the real logical chain (e.g. from column-projection pushdown).
    # `Project.infer_metadata()` (a one-to-one op with
    # can_modify_num_rows=False) only forwards `size_bytes` from its input
    # when that input actually has one -- this locks in that a V2 read's
    # estimate survives that hop, which is exactly what
    # `_try_estimate_output_bytes` (hash_shuffle.py) reads.
    for i in range(5):
        _write(
            tmp_path / f"f{i}.parquet", pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        )

    restore_ctx.use_datasource_v2 = True
    with pytest.warns(DeprecationWarning, match="`columns=` on `read_parquet`"):
        ds = ray.data.read_parquet(str(tmp_path), columns=["a"])

    from ray.data._internal.logical.operators.map_operator import Project

    dag = ds._logical_plan.dag
    assert isinstance(dag, Project)
    read_files_size_bytes = dag.input_dependencies[0].infer_metadata().size_bytes
    assert read_files_size_bytes is not None
    assert dag.infer_metadata().size_bytes == read_files_size_bytes


def test_read_parquet_v2_estimated_num_outputs(tmp_path, restore_ctx):
    # Regression test for a live SF1000 TPC-H hang: without a real
    # ``estimated_num_outputs()``, ``ReadFiles`` (via the inherited base
    # implementation, which walks up to ``ListFiles`` and finds no
    # ``num_outputs`` either) always returned ``None`` -- so any downstream
    # op that "matches upstream block count" when the caller doesn't specify
    # one (e.g. ``HashShufflingOperatorBase``'s ``target_num_partitions``)
    # fell back to a fixed default (200) regardless of how large the read
    # actually is. V1's equivalent (``Read._estimate_num_outputs()``) is
    # size-derived; this mirrors that formula exactly.
    import dataclasses
    import math

    _write(tmp_path / "f0.parquet", pa.table({"a": list(range(1000))}))
    restore_ctx.use_datasource_v2 = True
    ds = ray.data.read_parquet(str(tmp_path))
    read_files = ds._logical_plan.dag

    target_max_block_size = DataContext.get_current().target_max_block_size
    assert target_max_block_size is not None

    # Real estimate present -> ceil(size_bytes_estimate / target_max_block_size).
    # 10MB is deliberately small enough that ceil(...) stays under the
    # CPU-aware cap (max(read_op_min_num_blocks, 2*avail_cpus)) added in
    # ``_cap_against_cpu_ceiling`` under any realistic cluster size, so this
    # case exercises the uncapped formula. See
    # test_read_parquet_v2_estimated_num_outputs_cpu_cap for the cap itself.
    with_estimate = dataclasses.replace(read_files, size_bytes_estimate=10_000_000)
    assert with_estimate.estimated_num_outputs() == math.ceil(
        10_000_000 / target_max_block_size
    )

    # No estimate -> None, the safe fallback (unchanged from today).
    without_estimate = dataclasses.replace(read_files, size_bytes_estimate=None)
    assert without_estimate.estimated_num_outputs() is None

    # Empty dataset -> 0, not None (mirrors V1's zero-size edge case).
    empty = dataclasses.replace(read_files, size_bytes_estimate=0)
    assert empty.estimated_num_outputs() == 0

    # An explicit ``num_outputs`` always wins over the size-derived estimate.
    explicit = dataclasses.replace(
        read_files, num_outputs=42, size_bytes_estimate=10_000_000
    )
    assert explicit.estimated_num_outputs() == 42


def test_read_parquet_v2_estimated_num_outputs_cpu_cap(
    tmp_path, restore_ctx, monkeypatch
):
    # Regression test for a live release-test regression: on fixed-size
    # (non-autoscaling) clusters, the raw byte-size-driven estimate ran far
    # ahead of what CPU count would justify (SF1000 TPC-H q1:
    # num_partitions=2195 on V2 vs. 1000 on V1 for the same table/cluster),
    # making V2's hash-aggregate 1.24x-2.4x slower wall-clock than V1 on
    # those clusters (autoscaling clusters have slack to absorb it; fixed
    # ones don't). ``estimated_num_outputs()`` now caps the byte-size
    # estimate against V1's own ``max(read_op_min_num_blocks, 2*avail_cpus)``
    # ceiling from ``_autodetect_parallelism()``.
    import dataclasses
    import math

    _write(tmp_path / "f0.parquet", pa.table({"a": list(range(1000))}))
    restore_ctx.use_datasource_v2 = True
    ds = ray.data.read_parquet(str(tmp_path))
    read_files = ds._logical_plan.dag

    target_max_block_size = DataContext.get_current().target_max_block_size
    assert target_max_block_size is not None
    restore_ctx.read_op_min_num_blocks = 200
    # This test exercises the cap mechanism itself (default is now off --
    # see DEFAULT_READ_FILES_ESTIMATED_NUM_OUTPUTS_CAP_ENABLED), so enable it
    # explicitly rather than relying on the default.
    restore_ctx.read_files_estimated_num_outputs_cap_enabled = True

    huge = dataclasses.replace(
        read_files, size_bytes_estimate=10_000 * target_max_block_size
    )
    raw_huge_estimate = math.ceil(
        (10_000 * target_max_block_size) / target_max_block_size
    )

    def set_avail_cpus(n):
        monkeypatch.setattr(
            "ray.data._internal.util._estimate_available_parallelism", lambda: n
        )

    # Small byte estimate stays well under the ceiling -> uncapped.
    set_avail_cpus(4)
    small = dataclasses.replace(read_files, size_bytes_estimate=10_000_000)
    assert small.estimated_num_outputs() == math.ceil(
        10_000_000 / target_max_block_size
    )

    # Huge byte estimate + small avail_cpus -> capped to
    # max(read_op_min_num_blocks=200, 2*4) == 200.
    assert huge.estimated_num_outputs() == 200

    # Tiny-CPU cluster: the read_op_min_num_blocks floor still holds -- the
    # cap never suppresses the estimate below V1's own default floor.
    set_avail_cpus(1)
    assert huge.estimated_num_outputs() == 200

    # Large-CPU cluster: the CPU term (2*avail_cpus) binds once it exceeds
    # the read_op_min_num_blocks floor.
    set_avail_cpus(500)
    assert huge.estimated_num_outputs() == 1000

    # CPU detection failure -> fail open, raw uncapped estimate.
    def _raise():
        raise RuntimeError("no cluster")

    monkeypatch.setattr(
        "ray.data._internal.util._estimate_available_parallelism", _raise
    )
    assert huge.estimated_num_outputs() == raw_huge_estimate

    # Kill switch disabled -> uncapped even on a tiny-CPU cluster.
    set_avail_cpus(1)
    restore_ctx.read_files_estimated_num_outputs_cap_enabled = False
    assert huge.estimated_num_outputs() == raw_huge_estimate
    restore_ctx.read_files_estimated_num_outputs_cap_enabled = True


def test_read_parquet_v2_shuffle_files_randomizes_row_order(tmp_path, restore_ctx):
    # Regression test: FileAffinityPartitioner.finalize() used to sort emitted
    # partitions by path "for determinism," which silently discarded any
    # upstream shuffle for datasets small enough to never hit the
    # max_bucket_size overflow path (i.e. most small-file datasets) -- shuffle
    # had no effect at all, not even on the first execution. Fixed by
    # preserving arrival order instead of re-sorting.
    num_files = 15
    for i in range(num_files):
        _write(tmp_path / f"f{i}.parquet", pa.table({"file_id": [i]}))

    restore_ctx.use_datasource_v2 = True
    unshuffled_order = [
        r["file_id"] for r in ray.data.read_parquet(str(tmp_path)).iter_rows()
    ]
    shuffled_order = [
        r["file_id"]
        for r in ray.data.read_parquet(
            str(tmp_path), shuffle=FileShuffleConfig(seed=42)
        ).iter_rows()
    ]

    assert sorted(shuffled_order) == list(range(num_files))
    assert shuffled_order != unshuffled_order
    # The pre-fix bug produced exactly the lexicographic path order
    # (f0, f1, f10, f11, ..., f2, f3, ...), not a random permutation.
    assert shuffled_order != sorted(range(num_files), key=lambda i: f"f{i}.parquet")


def test_read_parquet_v2_shuffle_with_multi_row_group_files_reads_all_rows(
    tmp_path, restore_ctx
):
    # Regression test for a cursor[bot] finding: FileManifest.shuffle permutes
    # individual chunk rows, so a multi-row-group file's chunks can arrive at
    # FileAffinityPartitioner non-contiguously (unlike the single-chunk files
    # test_read_parquet_v2_shuffle_files_randomizes_row_order uses, which
    # can't exercise this). This only ever affected read efficiency (a
    # scattered file could fragment across extra read partitions), not
    # correctness, but assert end-to-end row-content correctness under
    # shuffle + multi-row-group files either way.
    num_files = 5
    rows_per_file = 20
    expected_ids = []
    for i in range(num_files):
        ids = list(range(i * rows_per_file, (i + 1) * rows_per_file))
        expected_ids.extend(ids)
        # row_group_size=5 -> 4 row groups per file, so each file contributes
        # multiple manifest rows (chunks) that shuffle can scatter.
        pq.write_table(
            pa.table({"id": ids}),
            str(tmp_path / f"f{i}.parquet"),
            row_group_size=5,
        )

    restore_ctx.use_datasource_v2 = True
    ds = ray.data.read_parquet(str(tmp_path), shuffle=FileShuffleConfig(seed=7))
    read_ids = sorted(r["id"] for r in ds.iter_rows())
    assert read_ids == sorted(expected_ids)


def test_read_parquet_v2_row_hashes_identical_across_footer_validation_flag(
    tmp_path, restore_ctx, monkeypatch
):
    # Regression test for the read-time footer-fetch elimination: row hashes
    # (which depend on each chunk's file-row offset) must be byte-identical
    # whether the offset comes from the listing-time-stamped value (default)
    # or a freshly re-derived footer read (validate_against_footer=True).
    # Multiple small, multi-row-group files land in one FileAffinityPartitioner
    # pack, exercising the coalesced-run offset lookup end to end.
    num_files = 5
    rows_per_file = 20
    for i in range(num_files):
        ids = list(range(i * rows_per_file, (i + 1) * rows_per_file))
        # row_group_size=5 -> 4 row groups per file, so chunks can be bundled.
        pq.write_table(
            pa.table({"id": ids}),
            str(tmp_path / f"f{i}.parquet"),
            row_group_size=5,
        )

    restore_ctx.use_datasource_v2 = True

    def _read_id_to_hash():
        ds = ray.data.read_parquet(str(tmp_path), include_row_hash=True)
        return {r["id"]: r["row_hash"] for r in ds.iter_rows()}

    default_result = _read_id_to_hash()
    assert len(default_result) == num_files * rows_per_file

    monkeypatch.setattr(
        DataContext.get_current(),
        "parquet_validate_chunk_ranges_at_read_time",
        True,
    )
    reverted_result = _read_id_to_hash()

    assert default_result == reverted_result


def test_read_parquet_v2_hive_partitioned(tmp_path, restore_ctx):
    for p in ["a", "b"]:
        d = tmp_path / f"color={p}"
        d.mkdir()
        _write(d / "data.parquet", pa.table({"x": [1, 2]}))

    restore_ctx.use_datasource_v2 = True
    ds = ray.data.read_parquet(str(tmp_path))
    schema = ds.schema()
    assert "x" in schema.names
    assert "color" in schema.names


def test_read_parquet_v2_include_paths(tmp_path, restore_ctx):
    _write(tmp_path / "data.parquet", pa.table({"a": [1]}))

    restore_ctx.use_datasource_v2 = True
    ds = ray.data.read_parquet(str(tmp_path), include_paths=True)
    schema = ds.schema()
    assert "path" in schema.names


def test_read_parquet_v2_include_row_hash(tmp_path, restore_ctx):
    _write(tmp_path / "data.parquet", pa.table({"a": [1, 2, 3]}))

    restore_ctx.use_datasource_v2 = True
    ds = ray.data.read_parquet(str(tmp_path), include_row_hash=True)
    schema = ds.schema()
    assert schema is not None
    assert "row_hash" in schema.names
    assert schema.types[schema.names.index("row_hash")] == pa.uint64()


def test_read_parquet_v2_columns_applies_select_columns(tmp_path, restore_ctx):
    from ray.data._internal.logical.operators.map_operator import Project

    _write(tmp_path / "data.parquet", pa.table({"a": [1], "b": [2]}))

    restore_ctx.use_datasource_v2 = True
    with pytest.warns(DeprecationWarning, match="`columns=` on `read_parquet`"):
        ds = ray.data.read_parquet(str(tmp_path), columns=["a"])

    # ``columns=`` is applied via ``ds.select_columns([...])``, which
    # wraps the ReadFiles op in a Project node.
    dag = ds._logical_plan.dag
    assert isinstance(dag, Project)
    assert [expr.name for expr in dag.exprs] == ["a"]
    assert isinstance(dag.input_dependencies[0], ReadFiles)


def test_read_parquet_v2_columns_with_include_paths_preserves_path(
    tmp_path, restore_ctx
):
    from ray.data._internal.logical.operators.map_operator import Project

    _write(tmp_path / "data.parquet", pa.table({"a": [1], "b": [2]}))

    restore_ctx.use_datasource_v2 = True
    with pytest.warns(DeprecationWarning, match="`columns=` on `read_parquet`"):
        ds = ray.data.read_parquet(str(tmp_path), columns=["a"], include_paths=True)

    dag = ds._logical_plan.dag
    assert isinstance(dag, Project)
    # V1 ``columns=[...]`` retained ``"path"`` implicitly when
    # ``include_paths=True``; the V2 path appends it to keep that
    # behavior.
    assert [expr.name for expr in dag.exprs] == ["a", "path"]


def test_read_parquet_v2_filter_raises(tmp_path, restore_ctx):
    import pyarrow.dataset as pds

    _write(tmp_path / "data.parquet", pa.table({"a": [1, 2, 3]}))

    restore_ctx.use_datasource_v2 = True
    with pytest.raises(ValueError, match="`filter=` on `read_parquet`"):
        ray.data.read_parquet(str(tmp_path), filter=pds.field("a") > 1)


def test_read_parquet_v2_dataset_kwargs_rejects_partitioning(tmp_path, restore_ctx):
    _write(tmp_path / "data.parquet", pa.table({"a": [1]}))

    restore_ctx.use_datasource_v2 = True
    with pytest.warns(DeprecationWarning, match="`dataset_kwargs`"):
        with pytest.raises(
            ValueError, match="'partitioning' parameter isn't supported"
        ):
            ray.data.read_parquet(
                str(tmp_path), dataset_kwargs={"partitioning": "hive"}
            )


def test_read_parquet_v2_dataset_kwargs_rejects_filters(tmp_path, restore_ctx):
    _write(tmp_path / "data.parquet", pa.table({"a": [1]}))

    restore_ctx.use_datasource_v2 = True
    with pytest.warns(DeprecationWarning, match="`dataset_kwargs`"):
        with pytest.raises(ValueError, match="Row filtering via 'filters'"):
            ray.data.read_parquet(
                str(tmp_path), dataset_kwargs={"filters": [("a", ">", 0)]}
            )


def test_read_parquet_v2_dataset_kwargs_threads_through_to_scanner(
    tmp_path, restore_ctx
):
    _write(tmp_path / "data.parquet", pa.table({"a": [1, 2, 3]}))

    restore_ctx.use_datasource_v2 = True
    with pytest.warns(DeprecationWarning, match="`dataset_kwargs`"):
        ds = ray.data.read_parquet(
            str(tmp_path),
            dataset_kwargs={
                "coerce_int96_timestamp_unit": "ms",
                "read_dictionary": ["a"],
            },
        )

    # ``read_dictionary`` is renamed to ``dictionary_columns`` to match
    # ``pds.ParquetFileFormat``; ``coerce_int96_timestamp_unit`` passes
    # through unchanged.
    read_files_op = ds._logical_plan.dag
    assert isinstance(read_files_op, ReadFiles)
    assert isinstance(read_files_op.scanner, ParquetScanner)
    assert read_files_op.scanner.parquet_format_kwargs == {
        "coerce_int96_timestamp_unit": "ms",
        "dictionary_columns": ["a"],
    }


def test_read_parquet_v2_empty_dir_raises(tmp_path, restore_ctx):
    restore_ctx.use_datasource_v2 = True
    with pytest.raises(ValueError, match="no files found"):
        ray.data.read_parquet(str(tmp_path))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
