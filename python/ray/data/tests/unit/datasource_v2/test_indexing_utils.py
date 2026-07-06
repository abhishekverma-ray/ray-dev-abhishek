"""Unit tests for ``indexing_utils.py``'s directory-listing helpers."""
import os

import pytest
from pyarrow.fs import LocalFileSystem

from ray.data._internal.datasource_v2.listing.indexing_utils import (
    _get_file_infos,
)


class _CountingFileSystem:
    """Duck-typed wrapper around a real filesystem that counts
    ``get_file_info`` calls. ``_get_file_infos``/``_expand_directory`` only
    ever call this one method on the ``filesystem`` argument, so a full
    ``pyarrow.fs.FileSystemHandler`` subclass (which requires implementing
    many unrelated abstract methods) isn't needed.
    """

    def __init__(self, real_filesystem):
        self._real = real_filesystem
        self.call_count = 0

    def get_file_info(self, arg):
        self.call_count += 1
        return self._real.get_file_info(arg)


def _touch(path):
    with open(path, "w") as f:
        f.write("x")


def _make_nested_tree(root):
    """3+ levels, multiple subdirs per level, mixed files/dirs, plus
    excluded (``.``/``_``-prefixed) entries at various positions."""
    expected = set()

    for top in ("a", "b"):
        top_dir = os.path.join(root, top)
        os.makedirs(top_dir)
        for mid in ("x", "y"):
            mid_dir = os.path.join(top_dir, mid)
            os.makedirs(mid_dir)
            for leaf in ("1", "2"):
                leaf_dir = os.path.join(mid_dir, leaf)
                os.makedirs(leaf_dir)
                p = os.path.join(leaf_dir, "file.txt")
                _touch(p)
                expected.add(p)
            p = os.path.join(mid_dir, "mid_file.txt")
            _touch(p)
            expected.add(p)
        p = os.path.join(top_dir, "top_file.txt")
        _touch(p)
        expected.add(p)

    # Top-level excluded directory: entirely pruned.
    hidden_dir = os.path.join(root, ".hidden")
    os.makedirs(hidden_dir)
    _touch(os.path.join(hidden_dir, "should_not_appear.txt"))

    underscore_dir = os.path.join(root, "_temp")
    os.makedirs(underscore_dir)
    _touch(os.path.join(underscore_dir, "should_not_appear_either.txt"))

    # Top-level excluded file.
    _touch(os.path.join(root, ".top_level_excluded.txt"))

    return expected


def test_expand_directory_matches_expected_set(tmp_path):
    root = str(tmp_path)
    expected_paths = _make_nested_tree(root)

    results = list(_get_file_infos(root, LocalFileSystem(), ignore_missing_path=False))
    result_paths = {path for path, _size in results}

    assert result_paths == expected_paths


def test_expand_directory_excludes_top_level_dot_and_underscore_prefixes(tmp_path):
    root = str(tmp_path)
    _make_nested_tree(root)

    results = list(_get_file_infos(root, LocalFileSystem(), ignore_missing_path=False))
    result_paths = {path for path, _size in results}

    assert not any(".hidden" in p for p in result_paths)
    assert not any("_temp" in p for p in result_paths)
    assert not any(".top_level_excluded.txt" in p for p in result_paths)


def test_expand_directory_does_not_exclude_nested_dot_prefix(tmp_path):
    # Regression: a `.`/`_`-prefixed entry NESTED under a normal parent is
    # NOT excluded -- the exclusion check only ever looked at the full
    # relative-from-root path, so it only prunes entries whose path relative
    # to the root itself starts with "." or "_". This is existing,
    # preserved behavior (not something this rewrite should change).
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "a", ".hidden_nested"))
    nested_file = os.path.join(root, "a", ".hidden_nested", "nested_file.txt")
    _touch(nested_file)
    normal_file = os.path.join(root, "a", "normal_file.txt")
    _touch(normal_file)

    results = list(_get_file_infos(root, LocalFileSystem(), ignore_missing_path=False))
    result_paths = {path for path, _size in results}

    assert result_paths == {nested_file, normal_file}


def test_ignore_missing_path_true_returns_empty(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    results = list(
        _get_file_infos(missing, LocalFileSystem(), ignore_missing_path=True)
    )
    assert results == []


def test_ignore_missing_path_false_raises(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        list(_get_file_infos(missing, LocalFileSystem(), ignore_missing_path=False))


def test_single_file_path_yields_one_entry(tmp_path):
    p = tmp_path / "single.txt"
    _touch(str(p))
    results = list(
        _get_file_infos(str(p), LocalFileSystem(), ignore_missing_path=False)
    )
    assert len(results) == 1
    assert results[0][0] == str(p)


def test_listing_call_count_is_not_proportional_to_subdirectory_count(tmp_path):
    """The actual point of the fix: a single recursive FileSelector call
    (O(1) logical listing operations) instead of one non-recursive call per
    subdirectory (O(N subdirectories)). Regression test that would have
    caught the original serial-per-directory-listing bug."""
    root = str(tmp_path)
    num_subdirs = 25
    for i in range(num_subdirs):
        d = os.path.join(root, f"class_{i:03d}")
        os.makedirs(d)
        _touch(os.path.join(d, "file.txt"))

    counting_fs = _CountingFileSystem(LocalFileSystem())
    results = list(_get_file_infos(root, counting_fs, ignore_missing_path=False))

    assert len(results) == num_subdirs
    # One call for the top-level `filesystem.get_file_info(path)` type-check
    # in `_get_file_infos`, plus exactly one recursive `FileSelector` call in
    # `_expand_directory` -- NOT one call per subdirectory (which would be
    # `num_subdirs + 1` or more under the old per-level-recursion approach).
    assert counting_fs.call_count == 2, (
        f"expected O(1) filesystem calls regardless of subdirectory count, "
        f"got {counting_fs.call_count} calls for {num_subdirs} subdirectories"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
