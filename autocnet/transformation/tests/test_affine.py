from plio.io.io_gdal import GeoDataset
import pytest

from autocnet.transformation import affine

def test_estimate_affine_transformation():
    gd_base = GeoDataset('tests/test_subpixel_match/B08_012650_1780_XN_02S046W.l1.cal.destriped.crop.cub')
    gd_match = GeoDataset('tests/test_subpixel_match/J04_046447_1777_XI_02S046W.l1.cal.destriped.crop.cub')
    affine_transform = affine.estimate_affine_transformation(gd_base,gd_match, 150, 150)
    assert affine_transform.rotation == pytest.approx(-0.0012609633370663982, 6)
    assert affine_transform.shear == pytest.approx(0.01262411827876344)
    assert affine_transform.scale[0] == pytest.approx(1.0, 6)
    assert affine_transform.scale[1] == pytest.approx(1.0000000000000002, 6) 
    assert affine_transform.translation[0] == pytest.approx(10.591885552140951, 6)
    assert affine_transform.translation[1] == pytest.approx(1.7377773467863733, 6)