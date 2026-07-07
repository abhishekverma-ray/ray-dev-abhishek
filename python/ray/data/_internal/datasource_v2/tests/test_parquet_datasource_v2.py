"""Unit tests for :class:`ParquetDatasourceV2`.

These tests exercise schema inference, scanner/estimator creation, and
include-paths schema augmentation against a local tmpdir — they do not
spin up Ray.
"""

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ray.data._internal.datasource_v2.chunkers.file_chunker import (
    ParquetFileChunker,
    ParquetFileChunkMetadata,
    WholeFileChunker,
    create_chunk_metadata,
)
from ray.data._internal.datasource_v2.chunkers.parquet_file_chunking_utils import (
    fragments_to_read_for_manifest,
)
from ray.data._internal.datasource_v2.listing.file_manifest import FileManifest
from ray.data._internal.datasource_v2.parquet_datasource_v2 import (
    ParquetDatasourceV2,
)
from ray.data._internal.datasource_v2.readers.in_memory_size_estimator import (
    ParquetFooterDerivedInMemorySizeEstimator,
)
from ray.data._internal.datasource_v2.readers.parquet_file_reader import (
    ParquetFileReader,
)
from ray.data._internal.datasource_v2.scanners.parquet_scanner import (
    ParquetScanner,
)
from ray.data.context import DataContext
from ray.data.datasource.partitioning import Partitioning, PartitionStyle


def _write_parquet(path: str, table: pa.Table) -> None:
    pq.write_table(table, path)


def _manifest_of(paths):
    sizes = [os.path.getsize(p) for p in paths]
    return FileManifest.construct_manifest(paths, sizes, [None] * len(paths))


def test_infer_schema_unpartitioned(tmp_path):
    file_path = tmp_path / "data.parquet"
    _write_parquet(str(file_path), pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}))

    datasource = ParquetDatasourceV2([str(file_path)])
    schema = datasource.infer_schema(_manifest_of([str(file_path)]))

    assert schema.names == ["a", "b"]
    assert schema.field("a").type == pa.int64()
    assert schema.field("b").type == pa.string()


def test_infer_schema_hive_partitioned(tmp_path):
    for part in ["a", "b"]:
        d = tmp_path / f"color={part}"
        d.mkdir()
        _write_parquet(str(d / "data.parquet"), pa.table({"x": [1, 2]}))

    first_file = str(tmp_path / "color=a" / "data.parquet")
    datasource = ParquetDatasourceV2(
        [str(tmp_path)], partitioning=Partitioning(PartitionStyle.HIVE)
    )
    schema = datasource.infer_schema(_manifest_of([first_file]))

    assert "x" in schema.names
    assert "color" in schema.names
    assert schema.field("color").type == pa.string()


def test_infer_schema_with_include_paths(tmp_path):
    file_path = tmp_path / "data.parquet"
    _write_parquet(str(file_path), pa.table({"a": [1, 2]}))

    datasource = ParquetDatasourceV2([str(file_path)], include_paths=True)
    schema = datasource.infer_schema(_manifest_of([str(file_path)]))

    assert "path" in schema.names
    assert schema.field("path").type == pa.string()


def test_infer_schema_returns_empty_schema_on_empty_manifest(tmp_path):
    datasource = ParquetDatasourceV2([str(tmp_path)])
    empty = FileManifest.construct_manifest([], [], [])
    schema = datasource.infer_schema(empty)
    assert schema.names == []


def test_create_scanner_returns_parquet_scanner(tmp_path):
    file_path = tmp_path / "data.parquet"
    _write_parquet(str(file_path), pa.table({"a": [1]}))

    datasource = ParquetDatasourceV2([str(file_path)])
    schema = datasource.infer_schema(_manifest_of([str(file_path)]))
    scanner = datasource.create_scanner(schema)

    assert isinstance(scanner, ParquetScanner)
    assert scanner.schema == schema


def test_get_size_estimator_returns_footer_derived(tmp_path):
    # V2 sizes partitions with the footer-derived, type-aware estimator: the
    # chunker stamps each chunk's decoded Arrow size onto its metadata at listing
    # time, and this estimator reads that hint (per-chunk ratio fallback).
    datasource = ParquetDatasourceV2([str(tmp_path)])
    assert isinstance(
        datasource.get_size_estimator(), ParquetFooterDerivedInMemorySizeEstimator
    )


def test_paths_and_filesystem_resolved(tmp_path):
    file_path = tmp_path / "data.parquet"
    _write_parquet(str(file_path), pa.table({"a": [1]}))

    datasource = ParquetDatasourceV2([str(file_path)])
    # _resolve_paths_and_filesystem produces a concrete filesystem even when
    # the caller passed None.
    assert datasource.filesystem is not None
    assert len(datasource.paths) == 1


def test_infer_schema_with_include_row_hash(tmp_path):
    file_path = tmp_path / "data.parquet"
    _write_parquet(str(file_path), pa.table({"a": [1, 2]}))

    datasource = ParquetDatasourceV2([str(file_path)], include_row_hash=True)
    schema = datasource.infer_schema(_manifest_of([str(file_path)]))

    assert "row_hash" in schema.names
    assert schema.field("row_hash").type == pa.uint64()


def test_infer_schema_with_include_row_hash_existing_column_promoted_to_uint64(
    tmp_path,
):
    file_path = tmp_path / "data.parquet"
    _write_parquet(str(file_path), pa.table({"val": [1, 2], "row_hash": [10, 20]}))

    datasource = ParquetDatasourceV2([str(file_path)], include_row_hash=True)
    schema = datasource.infer_schema(_manifest_of([str(file_path)]))

    assert schema.field("row_hash").type == pa.uint64()


def test_create_scanner_propagates_include_row_hash(tmp_path):
    file_path = tmp_path / "data.parquet"
    _write_parquet(str(file_path), pa.table({"a": [1]}))

    datasource = ParquetDatasourceV2([str(file_path)], include_row_hash=True)
    schema = datasource.infer_schema(_manifest_of([str(file_path)]))
    scanner = datasource.create_scanner(schema)

    assert scanner.include_row_hash is True


def test_nested_fallback_handles_schema_evolution(tmp_path, monkeypatch):
    """Regression: when the nested-type fallback fires on a fragment that
    lacks a filter-referenced column, the V2 reader must null-fill the
    missing column instead of letting pyarrow raise. Matches the
    scanner path, which null-fills via dataset-level schema pinning.
    """
    import pyarrow.dataset as pds

    from ray.data._internal.datasource import parquet_datasource
    from ray.data._internal.datasource_v2.readers.parquet_file_reader import (
        ParquetFileReader,
    )
    from ray.data.expressions import col

    _write_parquet(
        str(tmp_path / "with_b.parquet"),
        pa.table({"a": [1, 2, 3], "b": [10, 20, 30]}),
    )
    _write_parquet(
        str(tmp_path / "without_b.parquet"),
        pa.table({"a": [4, 5, 6]}),
    )

    unified_schema = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    predicate = col("b") > 15

    # Force the fallback path; the source-module attribute is what V2's
    # function-local import resolves to on each call.
    monkeypatch.setattr(
        parquet_datasource, "_needs_nested_type_fallback", lambda *a, **kw: True
    )

    reader = ParquetFileReader(
        columns=["a"], predicate=predicate, schema=unified_schema
    )
    dataset = pds.dataset(str(tmp_path), format="parquet", schema=unified_schema)
    scanner_kwargs = {
        "columns": ["a"],
        "filter": predicate.to_pyarrow(),
        "batch_size": None,
    }

    rows_by_fragment = {}
    for fragment in dataset.get_fragments():
        tables = list(reader._iter_fragment_tables(fragment, scanner_kwargs))
        rows_by_fragment[os.path.basename(fragment.path)] = sum(
            t.num_rows for t in tables
        )

    # with_b: rows where b > 15 → 2 rows (b=20, b=30)
    # without_b: b is null-filled → null > 15 is null → 0 rows
    assert rows_by_fragment == {"with_b.parquet": 2, "without_b.parquet": 0}


def test_datasource_defaults_to_parquet_file_chunker(tmp_path):
    """``ParquetDatasourceV2`` plugs ``ParquetFileChunker`` into its indexer."""
    file_path = tmp_path / "data.parquet"
    _write_parquet(str(file_path), pa.table({"a": [1, 2, 3]}))

    datasource = ParquetDatasourceV2([str(file_path)])
    indexer = datasource._get_file_indexer()
    assert isinstance(indexer.file_chunker, ParquetFileChunker)


def test_datasource_accepts_custom_chunker(tmp_path):
    """An explicit ``file_chunker`` override propagates to the indexer."""
    file_path = tmp_path / "data.parquet"
    _write_parquet(str(file_path), pa.table({"a": [1, 2, 3]}))

    custom = WholeFileChunker()
    datasource = ParquetDatasourceV2([str(file_path)], file_chunker=custom)
    indexer = datasource._get_file_indexer()
    assert indexer.file_chunker is custom


def _sample_and_scanner(datasource):
    """Mirror ``_read_datasource_v2``'s own sample+scanner construction."""
    from ray.data._internal.datasource_v2.listing.listing_utils import sample_files

    sample = sample_files(
        datasource._get_file_indexer(),
        datasource.paths,
        datasource.filesystem,
        [],
    )
    scanner = datasource.create_scanner(datasource.infer_schema(sample))
    return sample, scanner


def test_estimate_v2_read_size_bytes_returns_positive_estimate(tmp_path):
    from ray.data.read_api import _estimate_v2_read_size_bytes

    for i in range(5):
        _write_parquet(
            str(tmp_path / f"f{i}.parquet"), pa.table({"a": list(range(1000))})
        )

    datasource = ParquetDatasourceV2([str(tmp_path)])
    sample, scanner = _sample_and_scanner(datasource)

    estimate = _estimate_v2_read_size_bytes(datasource, sample, scanner)
    assert estimate is not None
    assert estimate > 0


def test_estimate_v2_read_size_bytes_uses_sampled_ratio_not_flat_default(tmp_path):
    # Regression test for a live SF1000 TPC-H hang: a flat on-disk-to-
    # in-memory ratio (5x) overshoots badly for fixed-width-heavy schemas
    # (ints, floats, decimals, dates) -- exactly what TPC-H's ``lineitem``
    # looks like -- inflating the estimate enough that a downstream
    # hash-aggregate's per-aggregator memory request exceeded total cluster
    # memory and hung. The estimate must use a real, sampled encoding ratio
    # (measured by actually decoding a sampled file, the same way V1's
    # ParquetDatasource does) instead of guessing a flat ratio.
    from ray.data._internal.datasource_v2.readers.in_memory_size_estimator import (
        PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT,
    )
    from ray.data.read_api import _estimate_v2_read_size_bytes

    # A fixed-width, mostly-numeric schema -- the real encoding ratio for
    # this kind of data is materially smaller than the flat 5x default.
    table = pa.table(
        {
            "a": list(range(100_000)),
            "b": [float(i) for i in range(100_000)],
        }
    )
    _write_parquet(str(tmp_path / "f0.parquet"), table)

    datasource = ParquetDatasourceV2([str(tmp_path)])
    sample, scanner = _sample_and_scanner(datasource)
    assert len(sample) == 1

    estimate = _estimate_v2_read_size_bytes(datasource, sample, scanner)
    flat_ratio_estimate = int(
        int(sample.file_sizes.sum()) * PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT
    )
    assert estimate < flat_ratio_estimate


def test_estimate_v2_read_size_bytes_extrapolates_over_full_listing(tmp_path):
    # The sampled ratio must be applied to the FULL listing's on-disk total,
    # not just the (possibly much smaller) sampled subset -- otherwise a
    # dataset with more files than the schema-inference sample cap would
    # silently under-count.
    from ray.data._internal.datasource_v2.readers.in_memory_size_estimator import (
        SamplingInMemorySizeEstimator,
    )
    from ray.data.read_api import _estimate_v2_read_size_bytes

    paths = [str(tmp_path / f"f{i}.parquet") for i in range(5)]
    for path in paths:
        _write_parquet(path, pa.table({"a": list(range(1000))}))

    datasource = ParquetDatasourceV2([str(tmp_path)])
    sample, scanner = _sample_and_scanner(datasource)

    estimate = _estimate_v2_read_size_bytes(datasource, sample, scanner)

    # Use real, filesystem-reported sizes throughout (the same source
    # _estimate_v2_read_size_bytes itself uses for both the ratio and the
    # full-listing total) -- a Parquet row group's chunk-derived byte size
    # excludes file-level footer/overhead bytes, so `sample.file_sizes`
    # doesn't exactly equal `os.path.getsize`. Pass the same `filesystem` the
    # real function does, so this uses the same footer-bounded, per-row-group
    # ratio measurement (not the whole-file fallback, which uses a slightly
    # different on-disk denominator and would make this comparison inexact).
    whole_file_sample = _manifest_of(paths)
    total_on_disk_bytes = int(whole_file_sample.file_sizes.sum())
    sample_estimator = SamplingInMemorySizeEstimator(
        scanner.create_reader(), filesystem=datasource.filesystem
    )
    sample_in_memory_bytes = float(
        sample_estimator.estimate_in_memory_sizes(whole_file_sample).sum()
    )
    ratio = sample_in_memory_bytes / float(whole_file_sample.file_sizes.sum())
    expected = int(total_on_disk_bytes * ratio)
    assert estimate == expected


def test_sampling_in_memory_size_estimator_averages_multiple_files(tmp_path):
    # Regression test for a review-confirmed accuracy gap (~3x, observed live
    # on a real TPC-H `lineitem` table): the estimator used to sample exactly
    # ONE file to establish the whole dataset's encoding ratio, even when many
    # more files were already available in the sample. V1's
    # ParquetDatasource._sample_fragments instead averages 2-10 files spread
    # across the manifest specifically to smooth out file-to-file compression
    # variance -- mirror that here.
    import numpy as np

    from ray.data._internal.datasource_v2.readers.in_memory_size_estimator import (
        SamplingInMemorySizeEstimator,
    )

    # File A: a single repeated value -- highly compressible (RLE/dictionary
    # encoding), so its encoding ratio (in-memory / on-disk) is large.
    path_a = str(tmp_path / "a.parquet")
    _write_parquet(path_a, pa.table({"a": [0] * 20_000}))

    # File B: high-entropy random values -- defeats RLE/dictionary/delta
    # encoding, so its encoding ratio is much closer to 1:1.
    path_b = str(tmp_path / "b.parquet")
    rng = np.random.default_rng(42)
    random_values = rng.integers(0, 2**62, size=20_000, dtype=np.int64).tolist()
    _write_parquet(path_b, pa.table({"a": random_values}))

    datasource = ParquetDatasourceV2([str(tmp_path)])

    def _ratio_for(paths):
        m = _manifest_of(paths)
        scanner = datasource.create_scanner(datasource.infer_schema(m))
        estimator = SamplingInMemorySizeEstimator(scanner.create_reader())
        in_memory_bytes = float(estimator.estimate_in_memory_sizes(m).sum())
        return in_memory_bytes / float(m.file_sizes.sum())

    ratio_a_only = _ratio_for([path_a])
    ratio_b_only = _ratio_for([path_b])
    # Test premise: file A must be meaningfully more compressible than B, or
    # this test isn't exercising anything.
    assert ratio_a_only > ratio_b_only * 1.5

    combined_manifest = _manifest_of([path_a, path_b])
    combined_scanner = datasource.create_scanner(
        datasource.infer_schema(combined_manifest)
    )
    combined_estimator = SamplingInMemorySizeEstimator(combined_scanner.create_reader())
    combined_estimator.estimate_in_memory_sizes(combined_manifest)
    averaged_ratio = combined_estimator._encoding_ratio

    # The averaged ratio must land strictly between the two individual
    # ratios -- not collapse to either single file's ratio alone, which is
    # what sampling only the first file (path_a) would produce.
    assert (
        min(ratio_a_only, ratio_b_only)
        < averaged_ratio
        < max(ratio_a_only, ratio_b_only)
    )


def test_sampling_in_memory_size_estimator_bounds_to_first_row_group(tmp_path):
    # Regression test for a review-confirmed accuracy bug: when a sampled
    # file's read naturally needed more than one batch (true for any
    # reasonably large file), the old implementation gave up computing a
    # real ratio and assumed decoded size == on-disk size (1:1 encoding) --
    # a large, systematic under-estimate, since Parquet data almost always
    # expands once decoded. Confirmed live: this made a real table's size
    # estimate come out ~3x too small. Bounding the sample read to the
    # file's first row group (its real, footer-derived on-disk size,
    # mirroring V1's own sampling) means a real ratio is always measured
    # -- unconditionally, regardless of how many batches that row group
    # itself decodes into (unlike the old code, which only computed a real
    # ratio if the *whole file* happened to decode in a single batch).
    import pyarrow.parquet as pq

    from ray.data._internal.datasource_v2.readers.in_memory_size_estimator import (
        SamplingInMemorySizeEstimator,
    )

    # Highly compressible data -- decoded size is much larger than on-disk
    # size, so a ~1.0 ratio (what the old "give up" fallback would produce)
    # is clearly distinguishable from the real one.
    file_path = tmp_path / "big.parquet"
    _write_parquet(str(file_path), pa.table({"a": [0] * 200_000}))

    datasource = ParquetDatasourceV2([str(tmp_path)])
    manifest = _manifest_of([str(file_path)])
    scanner = datasource.create_scanner(datasource.infer_schema(manifest))

    estimator = SamplingInMemorySizeEstimator(
        scanner.create_reader(), filesystem=datasource.filesystem
    )
    sizes = estimator.estimate_in_memory_sizes(manifest)
    ratio = float(sizes[0]) / float(manifest.file_sizes[0])

    # Independently compute the expected ratio: decode row group 0 for real
    # and divide by its own footer-reported *compressed* (on-disk) size --
    # not the whole file's on-disk size, which is what a correct
    # row-group-bounded estimate must use (this file has exactly one row
    # group, so "row group 0" and "the whole file" are the same data here).
    # ``RowGroupMetaData`` only exposes the *uncompressed* total_byte_size --
    # the on-disk size lives on each column chunk, so sum those.
    footer = pq.read_metadata(str(file_path))
    row_group_0 = footer.row_group(0)
    row_group_0_on_disk = sum(
        row_group_0.column(i).total_compressed_size
        for i in range(row_group_0.num_columns)
    )
    table = pq.read_table(str(file_path))
    expected_ratio = table.nbytes / row_group_0_on_disk

    assert ratio == pytest.approx(expected_ratio, rel=0.05)
    # And this must be a real, large ratio -- not the ~1.0 the old "give up"
    # fallback would have produced for a large, highly compressible file.
    assert ratio > 5.0


def test_estimate_v2_read_size_bytes_returns_none_on_listing_failure(
    tmp_path, monkeypatch
):
    # Must never raise -- this is a best-effort signal, not a correctness
    # requirement, and a failure here must not break the read.
    from ray.data import read_api

    _write_parquet(str(tmp_path / "f0.parquet"), pa.table({"a": [1, 2, 3]}))

    datasource = ParquetDatasourceV2([str(tmp_path)])
    sample, scanner = _sample_and_scanner(datasource)

    def _raise(*args, **kwargs):
        raise OSError("simulated listing failure")

    # `_estimate_v2_read_size_bytes` does a local import of `_get_file_infos`
    # from this module, so patching it here is what actually takes effect.
    import ray.data._internal.datasource_v2.listing.indexing_utils as indexing_utils

    monkeypatch.setattr(indexing_utils, "_get_file_infos", _raise)

    estimate = read_api._estimate_v2_read_size_bytes(datasource, sample, scanner)
    assert estimate is None


def _write_multi_row_group_parquet(path, num_rows: int, row_group_size: int):
    table = pa.table({"id": list(range(num_rows))})
    pq.write_table(table, path, row_group_size=row_group_size)
    return table


def test_estimate_v2_read_size_bytes_normalizes_chunked_sample_to_whole_file(tmp_path):
    # Regression test for a review-confirmed bug: when schema-inference
    # sampling uses the row-group-aware chunker (rather than a metadata-free
    # whole-file chunker), a single sample manifest row's on-disk size is
    # just one row group's size, not the whole file's. Dividing the whole
    # file's decoded in-memory size by one row group's on-disk size skews
    # the ratio by roughly the file's row-group count. The estimate must be
    # identical regardless of how many sample rows the chunker in use
    # happens to produce for this file.
    from ray.data._internal.datasource_v2.readers.in_memory_size_estimator import (
        SamplingInMemorySizeEstimator,
    )
    from ray.data.read_api import _estimate_v2_read_size_bytes

    file_path = tmp_path / "f0.parquet"
    _write_multi_row_group_parquet(str(file_path), num_rows=1000, row_group_size=50)

    # A tiny target_chunk_size forces the row-group-aware chunker to emit
    # one sample row per row group instead of bundling them all into one --
    # the premise this test exercises.
    datasource = ParquetDatasourceV2(
        [str(tmp_path)], file_chunker=ParquetFileChunker(target_chunk_size=1)
    )
    chunked_sample, scanner = _sample_and_scanner(datasource)
    assert len(chunked_sample) > 1

    estimate = _estimate_v2_read_size_bytes(datasource, chunked_sample, scanner)

    # Independently compute the correct ratio from a single, whole-file row
    # (what a metadata-free whole-file chunker's sample would look like).
    # Pass the same `filesystem` the real function does, so this uses the
    # same footer-bounded ratio measurement, not the whole-file fallback.
    whole_file_sample = _manifest_of([str(file_path)])
    sample_estimator = SamplingInMemorySizeEstimator(
        scanner.create_reader(), filesystem=datasource.filesystem
    )
    correct_in_memory_bytes = float(
        sample_estimator.estimate_in_memory_sizes(whole_file_sample).sum()
    )
    correct_ratio = correct_in_memory_bytes / float(whole_file_sample.file_sizes.sum())
    expected = int(os.path.getsize(str(file_path)) * correct_ratio)

    assert estimate == expected


def test_fragments_to_read_coalesces_sister_chunks(tmp_path):
    """Sister chunks of one file collapse into a single contiguous-run scan."""
    import pyarrow.dataset as pds

    file_path = str(tmp_path / "multi.parquet")
    _write_multi_row_group_parquet(file_path, num_rows=100, row_group_size=10)
    (fragment,) = pds.dataset(file_path, format="parquet").get_fragments()

    # Two adjacent row-group chunks of the same file: [0, 2) and [2, 4).
    # Each row group is 10 rows, so chunk_b's rows start at offset 20.
    chunk_a = create_chunk_metadata(
        ParquetFileChunkMetadata,
        row_group_start=0,
        row_group_end=2,
        in_memory_size=0,
        num_rows=20,
        row_offset=0,
    )
    chunk_b = create_chunk_metadata(
        ParquetFileChunkMetadata,
        row_group_start=2,
        row_group_end=4,
        in_memory_size=0,
        num_rows=20,
        row_offset=20,
    )
    frags = fragments_to_read_for_manifest(
        {fragment.path: fragment},
        [fragment.path, fragment.path],
        [chunk_a, chunk_b],
    )
    # One coalesced scan over [0, 4): a single open instead of one per chunk.
    assert len(frags) == 1
    sub, offset = frags[0]
    assert offset == 0
    assert len(sub.row_groups) == 4


def test_fragments_to_read_groups_by_file(tmp_path):
    """Chunks of different files map to one scan per file (no cross-file scan)."""
    import pyarrow.dataset as pds

    path_to_fragment, paths, metas = {}, [], []
    for name in ("a", "b"):
        p = str(tmp_path / f"{name}.parquet")
        _write_multi_row_group_parquet(p, num_rows=40, row_group_size=10)  # 4 rgs
        (fragment,) = pds.dataset(p, format="parquet").get_fragments()
        path_to_fragment[fragment.path] = fragment
        paths.append(fragment.path)
        metas.append(
            create_chunk_metadata(
                ParquetFileChunkMetadata,
                row_group_start=0,
                row_group_end=4,
                in_memory_size=0,
                num_rows=40,
                row_offset=0,
            )
        )
    frags = fragments_to_read_for_manifest(path_to_fragment, paths, metas)
    assert len(frags) == 2  # one sub-fragment per file
    assert {sub.path for sub, _ in frags} == set(paths)


def test_fragments_to_read_whole_file_chunk_passes_through(tmp_path):
    """A ``None`` (whole-file) chunk yields the full fragment at offset 0."""
    import pyarrow.dataset as pds

    file_path = str(tmp_path / "whole.parquet")
    _write_multi_row_group_parquet(file_path, num_rows=20, row_group_size=10)
    (fragment,) = pds.dataset(file_path, format="parquet").get_fragments()
    frags = fragments_to_read_for_manifest(
        {fragment.path: fragment}, [fragment.path], [None]
    )
    assert len(frags) == 1
    assert frags[0][1] == 0


def test_fragments_to_read_preserves_manifest_arrival_order(tmp_path):
    """A partition mixing a chunked file and a whole-file chunk must emit
    fragments in the manifest's arrival order, not whole-file-first
    regardless of order (the prior behavior)."""
    import pyarrow.dataset as pds

    chunked_path = str(tmp_path / "z_chunked.parquet")
    _write_multi_row_group_parquet(chunked_path, num_rows=20, row_group_size=10)
    (chunked_fragment,) = pds.dataset(chunked_path, format="parquet").get_fragments()

    whole_path = str(tmp_path / "a_whole.parquet")
    _write_multi_row_group_parquet(whole_path, num_rows=10, row_group_size=10)
    (whole_fragment,) = pds.dataset(whole_path, format="parquet").get_fragments()

    chunk = create_chunk_metadata(
        ParquetFileChunkMetadata,
        row_group_start=0,
        row_group_end=2,
        in_memory_size=0,
        num_rows=20,
        row_offset=0,
    )
    path_to_fragment = {
        chunked_fragment.path: chunked_fragment,
        whole_fragment.path: whole_fragment,
    }
    # Chunked file arrives FIRST, whole-file chunk SECOND, and the whole-file
    # path also sorts alphabetically before the chunked path -- so neither
    # arrival order nor alphabetical order would accidentally coincide with
    # the old "all whole-file fragments first" behavior.
    frags = fragments_to_read_for_manifest(
        path_to_fragment,
        [chunked_fragment.path, whole_fragment.path],
        [chunk, None],
    )
    assert [sub.path for sub, _ in frags] == [
        chunked_fragment.path,
        whole_fragment.path,
    ]


class _NoMetadataAccessFragment:
    """Wraps a real fragment; raises if ``.metadata`` is ever explicitly
    accessed at the Python level. ``.subset()`` is forwarded unmodified to the
    real fragment -- its own internal footer fetch (confirmed to bypass this
    Cython property entirely) is unaffected, so this proves the *default*
    ``fragments_to_read_for_manifest`` path never needs an explicit
    ``fragment.metadata`` access for offset computation."""

    def __init__(self, real_fragment):
        self._real = real_fragment
        self.path = real_fragment.path

    @property
    def metadata(self):
        raise AssertionError(
            "fragment.metadata was accessed -- the default path must not "
            "need a fresh footer read for row-offset computation."
        )

    def subset(self, **kwargs):
        return self._real.subset(**kwargs)


def test_fragments_to_read_uses_stamped_row_offset_not_footer(tmp_path):
    """Default path resolves offsets purely from stamped metadata -- no
    explicit ``fragment.metadata`` access."""
    import pyarrow.dataset as pds

    file_path = str(tmp_path / "multi.parquet")
    _write_multi_row_group_parquet(file_path, num_rows=100, row_group_size=10)
    (real_fragment,) = pds.dataset(file_path, format="parquet").get_fragments()
    guarded_fragment = _NoMetadataAccessFragment(real_fragment)

    chunk = create_chunk_metadata(
        ParquetFileChunkMetadata,
        row_group_start=2,
        row_group_end=4,
        in_memory_size=0,
        num_rows=20,
        row_offset=20,
    )
    frags = fragments_to_read_for_manifest(
        {guarded_fragment.path: guarded_fragment},
        [guarded_fragment.path],
        [chunk],
    )
    assert len(frags) == 1
    sub, offset = frags[0]
    assert offset == 20
    assert len(sub.row_groups) == 2


def test_fragments_to_read_offset_lookup_for_coalesced_runs_with_different_starts(
    tmp_path,
):
    """3 adjacent chunks (each with its own stamped ``row_offset``) coalesce
    into ONE contiguous run; the resulting sub-fragment must use the FIRST
    chunk's offset, not the last or a re-derived value."""
    import pyarrow.dataset as pds

    file_path = str(tmp_path / "multi.parquet")
    # 6 row groups x 10 rows each.
    _write_multi_row_group_parquet(file_path, num_rows=60, row_group_size=10)
    (fragment,) = pds.dataset(file_path, format="parquet").get_fragments()

    chunks = [
        create_chunk_metadata(
            ParquetFileChunkMetadata,
            row_group_start=start,
            row_group_end=start + 1,
            in_memory_size=0,
            num_rows=10,
            row_offset=start * 10,
        )
        for start in (2, 3, 4)
    ]
    frags = fragments_to_read_for_manifest(
        {fragment.path: fragment},
        [fragment.path] * 3,
        chunks,
    )
    assert len(frags) == 1  # all 3 adjacent chunks coalesce into one run
    sub, offset = frags[0]
    assert offset == 20  # the FIRST chunk's (row_group_start=2) row_offset
    assert len(sub.row_groups) == 3


def test_fragments_to_read_duplicate_identical_chunk_rows_same_start(tmp_path):
    """Two identical (row_group_start, row_offset) chunk rows for the same
    file don't crash and resolve to the (matching) offset."""
    import pyarrow.dataset as pds

    file_path = str(tmp_path / "multi.parquet")
    _write_multi_row_group_parquet(file_path, num_rows=40, row_group_size=10)
    (fragment,) = pds.dataset(file_path, format="parquet").get_fragments()

    chunk = create_chunk_metadata(
        ParquetFileChunkMetadata,
        row_group_start=1,
        row_group_end=2,
        in_memory_size=0,
        num_rows=10,
        row_offset=10,
    )
    frags = fragments_to_read_for_manifest(
        {fragment.path: fragment},
        [fragment.path, fragment.path],
        [chunk, chunk],
    )
    assert len(frags) == 1
    sub, offset = frags[0]
    assert offset == 10
    assert len(sub.row_groups) == 1


def test_fragments_to_read_validate_against_footer_flag_reverts(tmp_path):
    """``validate_against_footer=True`` ignores the stamped ``row_offset``
    entirely and re-derives it from a fresh footer read -- a stronger proof
    than "doesn't crash": deliberately stamp a WRONG offset and confirm the
    flag produces the CORRECT footer-derived value instead."""
    import pyarrow.dataset as pds

    file_path = str(tmp_path / "multi.parquet")
    _write_multi_row_group_parquet(file_path, num_rows=40, row_group_size=10)
    (fragment,) = pds.dataset(file_path, format="parquet").get_fragments()

    # Deliberately wrong: the true offset for row_group_start=2 is 20.
    chunk = create_chunk_metadata(
        ParquetFileChunkMetadata,
        row_group_start=2,
        row_group_end=3,
        in_memory_size=0,
        num_rows=10,
        row_offset=9999,
    )

    default_frags = fragments_to_read_for_manifest(
        {fragment.path: fragment}, [fragment.path], [chunk]
    )
    assert default_frags[0][1] == 9999  # trusts the (wrong) stamped value

    reverted_frags = fragments_to_read_for_manifest(
        {fragment.path: fragment},
        [fragment.path],
        [chunk],
        validate_against_footer=True,
    )
    assert reverted_frags[0][1] == 20  # re-derived correctly from the footer


def test_fragments_to_read_missing_file_raises_at_subset(tmp_path):
    """A file deleted between fragment discovery and
    ``fragments_to_read_for_manifest`` still raises -- the (unavoidable)
    footer fetch that used to happen via the removed explicit
    ``fragment.metadata`` call now happens inside ``.subset()`` instead, same
    exception surface, no loss of error detection."""
    import pyarrow.dataset as pds

    file_path = str(tmp_path / "vanishing.parquet")
    _write_multi_row_group_parquet(file_path, num_rows=20, row_group_size=10)
    (fragment,) = pds.dataset(file_path, format="parquet").get_fragments()
    os.remove(file_path)

    chunk = create_chunk_metadata(
        ParquetFileChunkMetadata,
        row_group_start=0,
        row_group_end=1,
        in_memory_size=0,
        num_rows=10,
        row_offset=0,
    )
    with pytest.raises(Exception):
        fragments_to_read_for_manifest(
            {fragment.path: fragment}, [fragment.path], [chunk]
        )


def _read_via_reader(reader, manifest):
    return list(reader.read(manifest))


def test_parquet_file_reader_reads_chunked_manifest(tmp_path):
    """End-to-end: a manifest with per-chunk rows is read into the same rows
    as a single whole-file manifest."""
    file_path = str(tmp_path / "data.parquet")
    expected_rows = 200
    _write_multi_row_group_parquet(file_path, num_rows=expected_rows, row_group_size=20)
    file_size = os.path.getsize(file_path)

    reader_whole = ParquetFileReader()
    whole_manifest = FileManifest.construct_manifest([file_path], [file_size], [None])
    whole_tables = _read_via_reader(reader_whole, whole_manifest)
    whole_rows = pa.concat_tables(whole_tables).column("id").to_pylist()

    # target_chunk_size=1 forces one chunk per row group.
    chunker = ParquetFileChunker(target_chunk_size=1)
    chunks = list(chunker.generate_chunk_metadatas(file_path, file_size))
    assert len(chunks) > 1, "test setup expects ParquetFileChunker to chunk"

    paths = [file_path] * len(chunks)
    chunk_metadatas = [md for md, _ in chunks]
    chunk_sizes = [sz for _, sz in chunks]
    chunked_manifest = FileManifest.construct_manifest(
        paths, chunk_sizes, chunk_metadatas
    )

    reader_chunked = ParquetFileReader()
    chunked_tables = _read_via_reader(reader_chunked, chunked_manifest)
    chunked_rows = pa.concat_tables(chunked_tables).column("id").to_pylist()

    assert sorted(chunked_rows) == sorted(whole_rows) == list(range(expected_rows))


def test_parquet_file_reader_reads_packed_multi_file_manifest(tmp_path):
    """End-to-end: a manifest packing rows from MULTIPLE distinct files (the
    shape FileAffinityPartitioner's packing produces) reads correctly -- all
    rows from all files, none dropped or duplicated. Exercises the real
    ParquetFileReader/fragments_to_read_for_manifest path, not just the
    partitioner in isolation (which is unit-tested separately in
    test_file_partitioners.py)."""
    paths, sizes, metas, expected_ids = [], [], [], []
    next_id = 0
    for i, num_rows in enumerate([5, 7, 3]):
        p = str(tmp_path / f"f{i}.parquet")
        ids = list(range(next_id, next_id + num_rows))
        pq.write_table(pa.table({"id": ids}), p)
        paths.append(p)
        sizes.append(os.path.getsize(p))
        metas.append(None)
        expected_ids.extend(ids)
        next_id += num_rows

    # Simulate a packed partition: one manifest whose rows span 3 distinct
    # files (FileAffinityPartitioner.finalize -> _flush_pending_pack would
    # produce exactly this shape via FileManifest.concat of per-file
    # manifests).
    packed_manifest = FileManifest.construct_manifest(paths, sizes, metas)

    reader = ParquetFileReader()
    tables = _read_via_reader(reader, packed_manifest)
    read_ids = sorted(pa.concat_tables(tables).column("id").to_pylist())
    assert read_ids == sorted(expected_ids)


def test_parquet_file_reader_chunked_row_hashes_are_unique(tmp_path):
    """Row hashes must remain unique across chunked sub-fragments of the
    same file.

    Regression: ``_read_fragments_sequential`` previously reseeded
    ``offset=0`` for every fragment. Since chunked sub-fragments share
    ``fragment.path``, ``_compute_row_hashes(path, 0, n)`` collided across
    row groups of the same file.
    """
    file_path = str(tmp_path / "data.parquet")
    expected_rows = 200
    _write_multi_row_group_parquet(file_path, num_rows=expected_rows, row_group_size=20)
    file_size = os.path.getsize(file_path)

    chunker = ParquetFileChunker(target_chunk_size=1)
    chunks = list(chunker.generate_chunk_metadatas(file_path, file_size))
    assert len(chunks) > 1, "test setup expects ParquetFileChunker to chunk"

    paths = [file_path] * len(chunks)
    chunk_metadatas = [md for md, _ in chunks]
    chunk_sizes = [sz for _, sz in chunks]
    chunked_manifest = FileManifest.construct_manifest(
        paths, chunk_sizes, chunk_metadatas
    )

    reader = ParquetFileReader(include_row_hash=True)
    chunked_tables = list(reader.read(chunked_manifest))
    hashes = pa.concat_tables(chunked_tables).column("row_hash").to_pylist()
    assert len(hashes) == expected_rows
    assert (
        len(set(hashes)) == expected_rows
    ), "row_hash must be unique across chunked sub-fragments of one file"


def test_parquet_file_reader_out_of_range_chunks_raise_by_default(tmp_path):
    """A hand-constructed out-of-range chunk now raises by default.

    The chunker never emits out-of-range ranges (they're computed from the
    same footer the reader sees), so this only matters for a hand-constructed
    manifest, or a file whose row-group count shrinks between listing and
    reading. Trading the old silent clamp-and-skip for a loud failure is an
    intentional, documented consequence of trusting listing-time-derived
    ranges by default (see ``_row_group_range_for_chunk``'s docstring) --
    ``validate_against_footer=True`` restores the old graceful behavior (see
    the companion test below).
    """
    file_path = str(tmp_path / "tiny.parquet")
    # 5 rows, single row group.
    _write_multi_row_group_parquet(file_path, num_rows=5, row_group_size=5)
    file_size = os.path.getsize(file_path)

    # Explicit range entirely beyond the file's one row group.
    out_of_range = create_chunk_metadata(
        ParquetFileChunkMetadata,
        row_group_start=3,
        row_group_end=4,
        in_memory_size=0,
        num_rows=0,
        row_offset=0,
    )
    manifest = FileManifest.construct_manifest([file_path], [file_size], [out_of_range])

    reader = ParquetFileReader()
    with pytest.raises(pa.ArrowIndexError):
        list(reader.read(manifest))


def test_parquet_file_reader_out_of_range_chunks_clamped_with_validate_flag(
    tmp_path, monkeypatch
):
    """``validate_against_footer=True`` (the rollback flag) restores the old
    defensive clamp-and-skip behavior for out-of-range chunk metadata."""
    monkeypatch.setattr(
        DataContext.get_current(),
        "parquet_validate_chunk_ranges_at_read_time",
        True,
    )
    file_path = str(tmp_path / "tiny.parquet")
    _write_multi_row_group_parquet(file_path, num_rows=5, row_group_size=5)
    file_size = os.path.getsize(file_path)

    out_of_range = create_chunk_metadata(
        ParquetFileChunkMetadata,
        row_group_start=3,
        row_group_end=4,
        in_memory_size=0,
        num_rows=0,
        row_offset=0,
    )
    manifest = FileManifest.construct_manifest([file_path], [file_size], [out_of_range])

    reader = ParquetFileReader()
    tables = list(reader.read(manifest))
    assert sum(t.num_rows for t in tables) == 0


class _StubFragment:
    def __init__(self, path: str):
        self.path = path


def test_resolve_num_read_workers_caps_at_distinct_files():
    """Per-task fragment-read concurrency keys on DISTINCT files, not fragment
    count: same-file sub-scans (file-affinity) stay sequential; cross-file
    partitions (round-robin) parallelize one worker per file, capped at the
    thread budget."""
    from ray.data._internal.datasource_v2.readers.file_reader import (
        _resolve_num_read_workers,
    )

    # File-affinity: many sub-fragments of ONE file -> sequential (1 worker).
    same_file = [(_StubFragment("a.parquet"), off) for off in (0, 2, 4, 6)]
    assert _resolve_num_read_workers(same_file, num_threads=4) == 1

    # Round-robin: 3 distinct files (one repeated) under a 4-thread budget -> 3.
    mixed = [
        (_StubFragment("a.parquet"), 0),
        (_StubFragment("b.parquet"), 0),
        (_StubFragment("c.parquet"), 0),
        (_StubFragment("a.parquet"), 5),
    ]
    assert _resolve_num_read_workers(mixed, num_threads=4) == 3

    # More distinct files than threads -> capped at the thread budget.
    many = [(_StubFragment(f"{c}.parquet"), 0) for c in "abcdef"]
    assert _resolve_num_read_workers(many, num_threads=4) == 4
