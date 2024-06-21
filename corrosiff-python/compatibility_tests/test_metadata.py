import pytest

import corrosiffpy

def test_metadata(siffreaders):
    corrosiff_sr, siffc_sr = siffreaders
    assert (
        (corrosiff_sr.get_experiment_timestamps()
        == siffc_sr.get_experiment_timestamps()).all()
    )

    assert (
        (corrosiff_sr.get_epoch_timestamps_laser()
        == siffc_sr.get_epoch_timestamps_laser()).all()
    )

    assert (
        (corrosiff_sr.get_epoch_timestamps_system()
        == siffc_sr.get_epoch_timestamps_system()).all()
    )

    assert (
        (corrosiff_sr.get_epoch_both()
        == siffc_sr.get_epoch_both()).all()
    )
