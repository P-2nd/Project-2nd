"""Tests for portable result-artifact paths."""

from pathlib import Path
import tempfile
import unittest

from utils import resolve_artifact_path


class ResultPathTests(unittest.TestCase):
    def test_relative_result_path_resolves_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            parquet_path = base_dir / "data/results/roc/model.parquet"
            parquet_path.parent.mkdir(parents=True)
            parquet_path.touch()

            resolved = resolve_artifact_path(
                "data/results/roc/model.parquet",
                base_dir=base_dir,
            )

            self.assertEqual(resolved, parquet_path)

    def test_legacy_windows_path_falls_back_to_local_roc_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            parquet_path = base_dir / "data/results/roc/model.parquet"
            parquet_path.parent.mkdir(parents=True)
            parquet_path.touch()

            resolved = resolve_artifact_path(
                r"C:\\SKN_33\\Project-2nd\\data\\results\\roc\\model.parquet",
                base_dir=base_dir,
            )

            self.assertEqual(resolved, parquet_path)
