"""Parquet file-level chunking helpers for DataSourceV2.

Maps planner chunk metadata to PyArrow ``ParquetFileFragment`` subsets for
parallel reads. :class:`ParquetFileChunkMetadata` carries an explicit half-open
row-group range (plus a ``row_offset``) computed at listing time from the
footer — by default the reader trusts this range and offset as-is, with no
read-time footer access for that purpose (see ``fragments_to_read_for_manifest``).

``fragments_to_read_for_manifest`` coalesces a partition's chunks **per file
into contiguous row-group runs**, so sister chunks of the same file (e.g. from
``FileAffinityPartitioner``) are read in a single scan — one file open,
one footer fetch (paid by ``fragment.subset()`` itself the first time it
touches an uncached fragment; unavoidable via PyArrow's public API, and no
worse than today), sequential I/O — instead of one scan per row group.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pyarrow.dataset as pds

from ray.data._internal.datasource_v2.chunkers.file_chunker import (
    ParquetFileChunkMetadata,
)


def _row_group_range_for_chunk(
    fragment: pds.ParquetFileFragment,
    chunk_metadata: ParquetFileChunkMetadata,
    *,
    validate_against_footer: bool = False,
) -> Optional[Tuple[int, int]]:
    """Resolve a chunk's half-open row-group range, or ``None`` if empty.

    By default trusts the listing-time-derived range as-is, with no footer
    access: the manifest is a snapshot of the files as they were listed, and
    within one Ray Data execution the same (immutable) Parquet files are read,
    so there is nothing to reconcile against. Set
    ``validate_against_footer=True`` (via
    ``RAY_DATA_PARQUET_VALIDATE_CHUNK_RANGES_AT_READ_TIME=1``) to restore the
    prior defensive clamp against ``fragment.metadata.num_row_groups`` — this
    re-introduces a read-time footer fetch.
    """
    start = chunk_metadata["row_group_start"]
    end = chunk_metadata["row_group_end"]
    if validate_against_footer:
        total_row_groups = fragment.metadata.num_row_groups
        start = min(start, total_row_groups)
        end = min(end, total_row_groups)
    return (start, end) if start < end else None


def _contiguous_runs(sorted_ids: List[int]) -> List[List[int]]:
    """Split a sorted list of row-group ids into maximal contiguous runs.

    e.g. ``[0, 1, 2, 5, 6] -> [[0, 1, 2], [5, 6]]``. Each run becomes one scan.
    """
    runs: List[List[int]] = []
    for rg in sorted_ids:
        if runs and rg == runs[-1][-1] + 1:
            runs[-1].append(rg)
        else:
            runs.append([rg])
    return runs


def fragments_to_read_for_manifest(
    path_to_fragment: Dict[str, pds.ParquetFileFragment],
    paths,
    chunk_metadatas,
    *,
    validate_against_footer: bool = False,
) -> List[Tuple[pds.ParquetFileFragment, int]]:
    """Map a partition's chunks to ``(sub_fragment, file_row_offset)`` scans,
    coalescing each file's row groups into **contiguous runs**.

    Sister chunks of the same file (consecutive row-group ranges, e.g. from
    ``FileAffinityPartitioner``) collapse into a single sub-fragment per run, so
    the reader opens the file once and streams those row groups sequentially.
    Whole-file chunks (``None`` metadata) pass through as the full fragment.

    By default (``validate_against_footer=False``), each run's file-row offset
    comes directly from the listing-time-stamped ``row_offset`` on whichever
    chunk contributed the run's first row-group id — no read-time footer
    access for this purpose (``fragment.subset()`` itself may still fetch the
    footer the first time it touches an uncached fragment; that cost is
    intrinsic to PyArrow's public API and is paid regardless of this flag —
    see the module docstring). Set ``validate_against_footer=True`` to fully
    revert to a fresh, footer-derived prefix sum instead (ignores the stamped
    ``row_offset`` entirely).
    """
    whole_file_paths: List[str] = []
    path_to_row_groups: Dict[str, Set[int]] = defaultdict(set)
    # Per-path map from a chunk's (post-clamp) row-group start to its
    # listing-time-stamped row_offset. A contiguous run's first row-group id
    # is always exactly some chunk's start (runs only break where a gap
    # exists in the unioned row-group id set, and ids only enter that set via
    # whole chunk ranges), so this lookup always hits for every run produced
    # below -- even when multiple chunks of one file are unioned together
    # (e.g. by ``FileAffinityPartitioner``).
    path_to_offset_by_start: Dict[str, Dict[int, int]] = defaultdict(dict)
    for path, chunk_metadata in zip(paths, chunk_metadatas):
        if chunk_metadata is None:
            whole_file_paths.append(path)
            continue
        rng = _row_group_range_for_chunk(
            path_to_fragment[path],
            chunk_metadata,
            validate_against_footer=validate_against_footer,
        )
        if rng is not None:
            path_to_row_groups[path].update(range(rng[0], rng[1]))
            path_to_offset_by_start[path][rng[0]] = chunk_metadata["row_offset"]

    fragments: List[Tuple[pds.ParquetFileFragment, int]] = []
    for path in whole_file_paths:
        fragments.append((path_to_fragment[path], 0))
    for path, row_groups in path_to_row_groups.items():
        fragment = path_to_fragment[path]
        if validate_against_footer:
            # Full revert: re-derive row_offsets from a fresh footer read,
            # ignoring the manifest's stamped row_offset entirely. Prefix sum
            # of per-row-group row counts: ``row_offsets[i]`` is the number of
            # rows in row groups ``[0, i)``.
            metadata = fragment.metadata
            row_offsets = [0] * (metadata.num_row_groups + 1)
            for i in range(metadata.num_row_groups):
                row_offsets[i + 1] = row_offsets[i] + metadata.row_group(i).num_rows
            for run in _contiguous_runs(sorted(row_groups)):
                fragments.append(
                    (fragment.subset(row_group_ids=run), row_offsets[run[0]])
                )
        else:
            offset_by_start = path_to_offset_by_start[path]
            for run in _contiguous_runs(sorted(row_groups)):
                fragments.append(
                    (fragment.subset(row_group_ids=run), offset_by_start[run[0]])
                )
    return fragments
