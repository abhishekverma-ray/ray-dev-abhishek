import logging
from typing import Iterable, Optional, Tuple

import pyarrow as pa
from pyarrow.fs import FileSelector, FileType

from ray.data.datasource.file_meta_provider import _handle_read_os_error

logger = logging.getLogger(__name__)


def _get_file_infos(
    path: str, filesystem: pa.fs.FileSystem, ignore_missing_path: bool
) -> Iterable[Tuple[str, Optional[int]]]:
    try:
        file_info = filesystem.get_file_info(path)
    except OSError as e:
        _handle_read_os_error(e, path)

    if file_info.type == FileType.Directory:
        yield from _expand_directory(path, filesystem, ignore_missing_path)
    elif file_info.type == FileType.File:
        yield (path, file_info.size)
    elif file_info.type == FileType.NotFound and ignore_missing_path:
        pass
    else:
        raise FileNotFoundError(path)


def _expand_directory(
    base_path: str,
    filesystem: pa.fs.FileSystem,
    ignore_missing_path: bool,
) -> Iterable[Tuple[str, Optional[int]]]:
    """Recursively list ``base_path``, pruning ``.``/``_``-prefixed entries.

    A single recursive ``FileSelector`` call (one logical, internally-
    paginated listing operation) rather than a manual walk that issues one
    non-recursive call per directory level -- the latter costs one network
    round-trip per subdirectory, which is serial and can dominate wall-clock
    for datasets with many subdirectories (e.g. per-class folders). Mirrors
    the equivalent, already-proven V1 helper at
    ``ray.data.datasource.file_meta_provider._expand_directory``.
    """
    exclude_prefixes = [".", "_"]

    selector = FileSelector(
        base_path, recursive=True, allow_not_found=ignore_missing_path
    )
    files = filesystem.get_file_info(selector)

    assert isinstance(files, list), type(files)
    base_dir = selector.base_dir

    out = []
    for file_ in files:
        if not file_.path.startswith(base_dir):
            continue

        relative = file_.path[len(base_dir) :].lstrip("/")
        if any(relative.startswith(prefix) for prefix in exclude_prefixes):
            continue

        if file_.type == FileType.File:
            out.append((file_.path, file_.size))
        elif file_.type == FileType.Directory:
            continue
        elif file_.type == FileType.UNKNOWN:
            logger.warning(f"Discovered file with unknown type: '{file_.path}'")
            continue
        else:
            assert file_.type == FileType.NotFound
            raise FileNotFoundError(file_.path)

    out.sort(key=lambda item: item[0])
    yield from out
