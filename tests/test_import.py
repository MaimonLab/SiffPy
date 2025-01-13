import pytest

def test_core_imports():
    from siffpy import SiffReader
    #from siffreadermodule import SiffIO
    import corrosiffpy

def test_registration_imports():
    try:
        import suite2p
    except ImportError:
        pytest.skip("Suite2p not installed")
    #import caiman