import pytest

@pytest.fixture
def test_core_imports():
    from siffpy import SiffReader
    from siffreadermodule import SiffIO

@pytest.fixture
def test_suite2p_imports():
    import suite2p