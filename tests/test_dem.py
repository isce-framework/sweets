from osgeo import osr
import pytest

from sweets import dem

BBOX = (-118.5, 33.6, -118.2, 33.9)


@pytest.fixture
def network_off():
    previous = osr.GetPROJEnableNetwork()
    osr.SetPROJEnableNetwork(False)
    yield
    osr.SetPROJEnableNetwork(previous)


def test_create_dem_refuses_when_the_geoid_cannot_be_applied(monkeypatch, tmp_path, network_off):
    # PROJ without a usable grid applies a no-op vertical shift
    monkeypatch.setattr(dem, "_egm2008_shift", lambda lon, lat: 0.0)
    called = []
    monkeypatch.setattr(dem.sardem.dem, "main", lambda **kwargs: called.append(kwargs))
    with pytest.raises(RuntimeError, match="EGM2008"):
        dem.create_dem(tmp_path / "dem.tif", BBOX)
    assert not called
    assert osr.GetPROJEnableNetwork() is False


def test_network_is_enabled_while_sardem_runs_then_restored(monkeypatch, tmp_path, network_off):
    monkeypatch.setattr(dem, "_egm2008_shift", lambda lon, lat: -36.0 if osr.GetPROJEnableNetwork() else 0.0)
    seen = []
    monkeypatch.setattr(dem.sardem.dem, "main", lambda **kwargs: seen.append(osr.GetPROJEnableNetwork()))
    dem.create_dem(tmp_path / "dem.tif", BBOX)
    assert seen == [True]
    assert osr.GetPROJEnableNetwork() is False
