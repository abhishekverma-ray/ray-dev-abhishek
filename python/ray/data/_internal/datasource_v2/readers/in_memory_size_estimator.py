import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ray.data._internal.datasource_v2.chunkers.file_chunker import (
    ParquetFileChunkMetadata,
    create_chunk_metadata,
)
from ray.data._internal.datasource_v2.listing.file_manifest import FileManifest
from ray.data._internal.datasource_v2.readers.file_reader import FileReader
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import BlockAccessor
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once

if TYPE_CHECKING:
    from pyarrow.fs import FileSystem

logger = logging.getLogger(__name__)


@DeveloperAPI
class InMemorySizeEstimator(ABC):
    @abstractmethod
    def estimate_in_memory_sizes(self, manifest: FileManifest) -> np.ndarray:
        """Estimate the in-memory sizes of the paths in the given manifest.

        Some `FilePartitioner` implementations use this method to ensure that each
        read task receives an appropriate amount of data. To ensure that file listing
        is efficient, this method must be cheap to call, on average.

        Args:
            manifest: A manifest containing the paths and on-disk sizes of the files.

        Returns:
            The estimated in-memory sizes of the data in bytes.
        """
        ...


# Sampling parameters for SamplingInMemorySizeEstimator's encoding-ratio
# estimate -- mirrors V1's ParquetDatasource._sample_fragments
# (parquet_datasource.py): average a handful of files spread evenly across
# the manifest instead of trusting a single file, to smooth out
# file-to-file compression/encoding variance. A single-file sample can
# diverge substantially from the true dataset-wide ratio -- confirmed via a
# live A/B on the same table: V1's multi-file-averaged estimate was ~3x
# V2's single-file one.
_ENCODING_RATIO_SAMPLING_RATIO = 0.01
_ENCODING_RATIO_MIN_NUM_SAMPLES = 2
_ENCODING_RATIO_MAX_NUM_SAMPLES = 10


@DeveloperAPI
class SamplingInMemorySizeEstimator(InMemorySizeEstimator):
    """Estimates in-memory sizes by reading files.

    This class estimates the in-memory size of files by multiplying the on-disk
    size by an estimated encoding ratio, averaged over a handful of files spread
    across the manifest (mirrors V1's ``ParquetDatasource._sample_fragments``) to
    smooth out file-to-file compression/encoding variance. If an instance hasn't
    estimated an encoding ratio yet, it'll sample files to estimate it. Otherwise,
    it'll use the previously estimated encoding ratio.

    When ``filesystem`` is provided, each sampled file's ratio is measured from
    just its first Parquet row group (bounded, footer-derived on-disk size),
    mirroring V1's ``_fetch_parquet_file_info``: this always yields a real,
    measured ratio, no matter how large the rest of the file is. Without a
    ``filesystem`` (or if the footer can't be read), falls back to reading the
    file via ``reader`` directly, which only produces a real ratio when the file
    happens to decode in a single batch -- for a large file that needs more than
    one, it assumes a 1:1 ratio rather than reading the whole thing (confirmed via
    a live A/B: this silently under-estimated a real table's size by ~3x).
    """

    def __init__(self, reader: "FileReader", filesystem: Optional["FileSystem"] = None):
        self._reader = reader
        self._filesystem = filesystem

        self._encoding_ratio = None

    def estimate_in_memory_sizes(self, manifest: FileManifest) -> np.ndarray:
        assert np.all(manifest.file_sizes >= 0)

        if self._encoding_ratio is None:
            # Estimating the encoding ratio can be expensive since it requires
            # reading files. So, we only estimate it if we don't already have one.
            self._encoding_ratio = self._estimate_encoding_ratio(manifest)

        if self._encoding_ratio is None:
            # If we couldn't estimate the encoding ratio, assume a 1:1 encoding ratio.
            return manifest.file_sizes
        else:
            return manifest.file_sizes * self._encoding_ratio

    def _estimate_encoding_ratio(self, manifest: FileManifest) -> Optional[float]:
        """Estimate the dataset's encoding ratio (in-memory size / on-disk size)
        by sampling a handful of files spread evenly across ``manifest`` and
        averaging their individual ratios.

        Args:
            manifest: The manifest to sample files from.

        Returns:
            The estimated encoding ratio, or ``None`` if no sampled file yielded
            a usable ratio.
        """
        n = len(manifest)
        if n == 0:
            return None

        target_num_samples = math.ceil(n * _ENCODING_RATIO_SAMPLING_RATIO)
        target_num_samples = max(
            min(target_num_samples, _ENCODING_RATIO_MAX_NUM_SAMPLES),
            _ENCODING_RATIO_MIN_NUM_SAMPLES,
        )
        # Make sure the number of samples doesn't exceed the number of files.
        target_num_samples = min(target_num_samples, n)
        pivots = np.linspace(0, n - 1, target_num_samples).astype(int)

        ratios: List[float] = []
        for idx in pivots.tolist():
            ratio = self._estimate_file_encoding_ratio(
                manifest.paths[idx], manifest.file_sizes[idx]
            )
            if ratio is not None:
                ratios.append(ratio)

        if not ratios:
            return None
        return float(np.mean(ratios))

    def _estimate_file_encoding_ratio(
        self,
        path: str,
        file_size: int,
    ) -> Optional[float]:
        """
        Estimate the encoding ratio (in-memory size / on-disk size) for a single
        file.

        Args:
            path: The path to the file.
            file_size: The on-disk size of the file/chunk in bytes.

        Returns:
            The estimated encoding ratio of the file, or `None` if the ratio can't
            be estimated.
        """
        # If the file is empty, we can't estimate the encoding ratio.
        if not file_size:
            return None

        row_group_on_disk_size = self._first_row_group_on_disk_size(path)
        if row_group_on_disk_size:
            ratio = self._estimate_ratio_from_row_group(
                path, file_size, row_group_on_disk_size
            )
            if ratio is not None:
                return ratio
        return self._estimate_ratio_from_whole_file(path, file_size)

    def _first_row_group_on_disk_size(self, path: str) -> Optional[int]:
        """Return the first row group's on-disk (compressed) byte size from the
        Parquet footer, or ``None`` if it can't be determined (no filesystem
        configured, the footer can't be read, or the file has no row groups).

        ``RowGroupMetaData`` exposes only the *uncompressed* ``total_byte_size``
        -- the on-disk size lives on each ``ColumnChunkMetaData``, so sum the
        per-column compressed sizes (mirrors ``ParquetFileChunker``'s footer
        loop, which needs this same on-disk figure for the same reason).
        """
        if self._filesystem is None:
            return None
        try:
            import pyarrow.parquet as pq

            metadata = pq.read_metadata(path, filesystem=self._filesystem)
            if metadata.num_row_groups == 0:
                return None
            row_group = metadata.row_group(0)
            compressed_size = sum(
                row_group.column(i).total_compressed_size
                for i in range(row_group.num_columns)
            )
            return compressed_size or None
        except Exception as e:
            logger.debug(
                "Failed to read footer for '%s' to bound the encoding-ratio "
                "sample read: %s",
                path,
                e,
            )
            return None

    def _estimate_ratio_from_row_group(
        self,
        path: str,
        file_size: int,
        row_group_on_disk_size: int,
    ) -> Optional[float]:
        """Estimate the encoding ratio from just the file's first row group.

        Bounding the read to one row group (rather than the whole file) means
        this always yields a real, measured ratio, no matter how large the rest
        of the file is or how many batches the row group itself decodes into --
        unlike ``_estimate_ratio_from_whole_file``, which gives up and assumes a
        1:1 ratio the moment a file needs more than one batch.
        """
        chunk_metadata = create_chunk_metadata(
            ParquetFileChunkMetadata,
            row_group_start=0,
            row_group_end=1,
            # Not used by the read path itself (only row_group_start/end are);
            # these are placeholders for the manifest's benefit.
            in_memory_size=0,
            num_rows=0,
            row_offset=0,
        )
        manifest = FileManifest.construct_manifest(
            [path],
            [file_size],
            [chunk_metadata],
        )
        builder = DelegatingBlockBuilder()
        has_data = False
        for batch in self._reader.read(manifest):
            builder.add_batch(batch)
            has_data = True
        if not has_data:
            return None
        in_memory_size = BlockAccessor.for_block(builder.build()).size_bytes()
        if not in_memory_size:
            return None
        return in_memory_size / row_group_on_disk_size

    def _estimate_ratio_from_whole_file(
        self,
        path: str,
        file_size: int,
    ) -> Optional[float]:
        """Fallback used when the footer can't be read (e.g. no ``filesystem``
        configured): read the whole file via ``reader`` directly. Only yields a
        real ratio if the file decodes in a single batch; otherwise assumes a
        1:1 ratio rather than reading the whole (potentially large) file.
        """
        # Use ``None`` chunk metadata: the size estimator reads the file whole
        # to estimate the encoding ratio; chunk-level splitting is irrelevant here.
        manifest = FileManifest.construct_manifest(
            [path],
            [file_size],
            [None],
        )
        batches = self._reader.read(manifest)

        try:
            first_batch = next(batches)
        except StopIteration:
            # If there's no data, we can't estimate the encoding ratio.
            return None

        try:
            # Try to read a second batch. If it succeeds, it means the file contains
            # multiple batches.
            next(batches)
        except StopIteration:
            # Each file contains exactly one batch.
            builder = DelegatingBlockBuilder()
            builder.add_batch(first_batch)
            block = builder.build()

            in_memory_size = BlockAccessor.for_block(block).size_bytes()
        else:
            # Each file contains multiple batches.
            #
            # NOTE: To avoid reading the entire file to estimate the encoding ratio,
            # we assume the file is 1:1 encoded. We can't return `None` because if
            # all files contain multiple batches, then we'd try to re-estimate the
            # encoding ratio for every file, and that'd be very expensive.
            in_memory_size = file_size

        return in_memory_size / file_size


# Default Parquet encoding ratio: in-memory is ~5x on-disk size.
# Parquet uses columnar compression and encoding, so Arrow in-memory
# representation is significantly larger than the on-disk format.
PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT = 5


@DeveloperAPI
class ParquetInMemorySizeEstimator(InMemorySizeEstimator):
    """Estimates in-memory sizes for Parquet files using a fixed encoding ratio.

    Parquet files are typically much smaller on disk than in memory due to
    columnar compression and encoding. This estimator applies a constant
    ratio (default 5x) to avoid the overhead of reading file metadata or
    sampling data, which can be slow for Parquet files and hurt startup time.
    """

    def __init__(self, encoding_ratio: float = PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT):
        self._encoding_ratio = encoding_ratio

    def estimate_in_memory_sizes(self, manifest: FileManifest) -> np.ndarray:
        return self._encoding_ratio * manifest.file_sizes


def _as_finite_float(value) -> float:
    """Coerce ``value`` to a float, mapping ``None`` and ``NaN`` to ``0.0``.

    File sizes can be ``None`` (e.g. ``HTTPFileSystem``, which doesn't report
    sizes) or surface as ``NaN`` from a nullable size column; either would make
    a downstream ``float(...)`` raise.
    """
    if value is None or value != value:  # ``value != value`` is True only for NaN
        return 0.0
    return float(value)


@DeveloperAPI
class ParquetFooterDerivedInMemorySizeEstimator(InMemorySizeEstimator):
    """Parquet-specific estimator that reads the per-chunk footer-derived hint.

    The row-group-aware ``ParquetFileChunker`` reads each Parquet file's footer at
    listing time and stamps a type-aware Arrow in-memory estimate onto each
    chunk's metadata under the ``in_memory_size`` key -- assigned in
    ``ParquetFileChunker.generate_chunk_metadatas`` (its ``_emit`` helper) and
    carried through the manifest's chunk-metadata column. This estimator reads
    that hint, so partition sizing reflects each chunk's actual columns --
    absorbing cross-file compression and encoding variance -- instead of a single
    global on-disk × encoding-ratio guess.

    Chunks without a hint (whole-file fallback on a corrupt/empty footer, or
    non-Parquet inputs) -- or with a hint of exactly ``0`` (a suspicious
    footer-accounting corner case for a chunk with real on-disk bytes, e.g. an
    all-dictionary/struct schema) -- fall back to ``on_disk_size ×
    fallback_ratio``, the constant-ratio behavior, so mixed manifests are
    handled row by row and a real chunk never contributes 0 weight to
    ``FileAffinityPartitioner``'s size-cap flush.
    """

    def __init__(self, fallback_ratio: float = PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT):
        self._fallback_ratio = fallback_ratio

    def estimate_in_memory_sizes(self, manifest: FileManifest) -> np.ndarray:
        file_sizes = manifest.file_sizes
        chunk_metadatas = manifest.file_chunk_metadatas
        out = np.empty(len(file_sizes), dtype=np.float64)
        for i in range(len(file_sizes)):
            md = chunk_metadatas[i]
            hint = md.get("in_memory_size") if isinstance(md, dict) else None
            # A hint of exactly 0 is treated the same as a missing hint: a
            # chunk's on-disk bytes (file_sizes[i], per-chunk here) are
            # essentially never 0 for a real row-group chunk, so a 0 hint on
            # such a chunk is a suspicious footer-accounting corner case
            # (e.g. an all-dictionary/struct schema whose uncompressed bytes
            # weren't attributed), not a genuine zero-byte chunk. Falling
            # through to the ratio-based estimate avoids stamping a 0 weight
            # onto real data, which would let it skip FileAffinityPartitioner's
            # max_bucket_size flush entirely. A truly empty chunk (0 on-disk
            # bytes too) still estimates to 0 via the fallback, so this is a
            # no-op for the legitimate zero case.
            if hint:
                out[i] = float(hint)
            else:
                path = manifest.paths[i]
                if log_once(f"parquet_footer_hint_missing_v2:{path}"):
                    logger.debug(
                        "No usable footer-derived in_memory_size hint for '%s' "
                        "(missing or 0); falling back to on_disk_size * %s.",
                        path,
                        self._fallback_ratio,
                    )
                out[i] = _as_finite_float(file_sizes[i]) * self._fallback_ratio
        return out
