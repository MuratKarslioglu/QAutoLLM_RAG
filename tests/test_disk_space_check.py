from pathlib import Path
from unittest.mock import patch

import pytest

from qautollm.utils.profiling import disk_probe
from scripts.run_experiment import _check_disk_space_for_generator_download


def test_disk_probe_reports_plausible_values(tmp_path: Path):
    usage = disk_probe(tmp_path)
    assert usage["total_gb"] > 0
    assert usage["free_gb"] >= 0
    assert usage["used_gb"] >= 0
    assert usage["free_gb"] <= usage["total_gb"]


def test_disk_space_check_raises_when_below_threshold():
    with patch("scripts.run_experiment.disk_probe", return_value={"free_gb": 5.0}):
        with pytest.raises(RuntimeError, match="only 5.0GB free"):
            _check_disk_space_for_generator_download()


def test_disk_space_check_passes_when_above_threshold():
    with patch("scripts.run_experiment.disk_probe", return_value={"free_gb": 50.0}):
        _check_disk_space_for_generator_download()  # must not raise
