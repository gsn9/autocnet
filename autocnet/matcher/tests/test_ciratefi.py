import math
import unittest
import logging
import re

import numpy as np
from imageio import imread
from scipy.ndimage.interpolation import rotate

from autocnet.examples import get_path
from autocnet.transformation import roi
from .. import ciratefi

import pytest

# Can be parameterized for more exhaustive tests
upsampling = 10
alpha = math.pi/2
cifi_thresh = 90
rafi_thresh = 90
tefi_thresh = 100
use_percentile = True
radii = list(range(1, 3))

@pytest.fixture
def img():
    return imread(get_path('AS15-M-0298_SML.png'), as_gray=True)

@pytest.fixture
def img_coord():
    return 482.09783936, 652.40679932

@pytest.fixture
def template(img, img_coord):
    coord_x, coord_y = img_coord
    template= roi.Roi(img, coord_x, coord_y, 5, 5).clip()
    template = rotate(template, 90)
    return template

@pytest.fixture
def search():
    coord_x, coord_y = (482.09783936, 652.40679932)
    img = imread(get_path('AS15-M-0298_SML.png'), as_gray=True)
    search = roi.Roi(img, coord_x, coord_y, 21, 21).clip()
    return search

@pytest.fixture
def offset_template(img, img_coord):
    coord_x, coord_y = img_coord
    coord_x += 1
    coord_y += 1
    offset_template = roi.Roi(img, coord_x, coord_y, 5, 5).clip()
    return offset_template

def test_cifi_radii_too_large(template, search, caplog):
    # check all logs
    ciratefi.cifi(template, search, 1.0, radii=[100], use_percentile=False)
    num_pattern = '\d+'
    captured_log = caplog.records[0].getMessage()
    match = re.findall(num_pattern, captured_log)
    assert (f'Max Radii is larger than original template, '\
        f'this may produce sub-par results.Max radii: {match[0]} max template dimension: {match[1]}'\
             == caplog.records[0].getMessage())


def test_cifi_bounds_error(template, search):
    with pytest.raises(ValueError), pytest.warns(UserWarning):
        ciratefi.cifi(template, search, -1.1, use_percentile=False)

def test_cifi_radii_none_error(template, search):
    with pytest.raises(ValueError):
        ciratefi.cifi(template, search, 90, radii=None)

def test_cifi_scales_none_error(template, search):
    with pytest.raises(ValueError):
        ciratefi.cifi(template, search, 90, scales=None)

def test_cifi_template_too_large_error(template, search):
    with pytest.raises(ValueError):
        ciratefi.cifi(search,template, 90, scales=None)

@pytest.mark.parametrize('cifi_thresh, radii', [(90,list(range(1, 3)))])
@pytest.mark.filterwarnings('ignore::UserWarning')  # skimage deprecation warnings to move to new defaults
def test_cifi(template, search, cifi_thresh, radii):
    pixels, scales = ciratefi.cifi(template, search, thresh=cifi_thresh,
                                   radii=radii, use_percentile=True)

    assert search.shape == scales.shape
    assert (np.floor(search.shape[0]/2), np.floor(search.shape[1]/2)) in pixels
    assert pixels.size in range(0,search.size)

@pytest.mark.filterwarnings('ignore::UserWarning')  # skimage deprecation warnings to move to new defaults
def test_rafi_warning(template, search,caplog):
    rafi_pixels = [(10, 10)]
    rafi_scales = np.ones(search.shape, dtype=float)
    ciratefi.rafi(template, search, rafi_pixels,
            rafi_scales, thresh=1, radii=[100],
            use_percentile=False)

def test_rafi_bounds_error(template, search):
    rafi_pixels = [(10, 10)]
    rafi_scales = np.ones(search.shape, dtype=float)
    with pytest.raises(ValueError) as f, pytest.warns(UserWarning) as g:
        ciratefi.rafi(template, search, rafi_pixels, rafi_scales, -1.1, use_percentile=False)

def test_rafi_radii_list_none_error(template, search):
    rafi_pixels = [(10, 10)]
    with pytest.raises(ValueError):
        ciratefi.rafi(search, template, rafi_pixels, -1.1, radii=None)

def test_rafi_pixel_list_error(template, search):
    rafi_pixels = []
    rafi_scales = np.ones(search.shape, dtype=float)
    with pytest.raises(ValueError):
        ciratefi.rafi(template, search, rafi_pixels, rafi_scales)

def test_rafi_scales_list_error(template, search):
    rafi_pixels = [(10, 10)]
    with pytest.raises(ValueError):
        ciratefi.rafi(template, search, rafi_pixels, None)

def test_rafi_template_bigger_error(template, search):
    rafi_pixels = [(10, 10)]
    rafi_scales = np.ones(search.shape, dtype=float)
    with pytest.raises(ValueError):
        ciratefi.rafi(search, template, rafi_pixels,rafi_scales)

def test_rafi_shape_mismatch(template, search):
        rafi_pixels = [(10, 10)]
        rafi_scales = np.ones(search.shape, dtype=float)[:10]
        with pytest.raises(ValueError):
            ciratefi.rafi(template, search, rafi_pixels, rafi_scales)

@pytest.mark.parametrize("rafi_thresh, radii, alpha", [(90, list(range(1, 3)),math.pi/2)])
@pytest.mark.filterwarnings('ignore::UserWarning')  # skimage deprecation warnings to move to new defaults
def test_rafi(template, search, rafi_thresh, radii, alpha):
    rafi_pixels = [(10, 10)]
    rafi_scales = np.ones(search.shape, dtype=float)
    pixels, scales = ciratefi.rafi(template, search, rafi_pixels, rafi_scales,
                                   thresh=rafi_thresh, radii=radii, use_percentile=True,
                                   alpha=alpha)

    assert (np.floor(search.shape[0]/4), np.floor(search.shape[1]/4)) in pixels
    assert pixels.size in range(0, search.size)

# Alternate approach to the more verbose tests above - this tests all combinations
@pytest.mark.parametrize("tefi_pixels", [[(10,10)], None])
@pytest.mark.parametrize("tefi_scales", [np.ones((42,42), dtype=float),
                                         None,
                                         np.ones((42,42), dtype=float)[:10]])
@pytest.mark.parametrize("tefi_angles", [[3.14159265], None])
@pytest.mark.parametrize("thresh", [-1.1, 90])
@pytest.mark.parametrize("reverse", [False, True])
def test_tefi_errors(template, search, tefi_pixels, tefi_scales, tefi_angles, thresh, reverse):
    with pytest.raises(ValueError):
        if reverse:
            template, search = search, template
        ciratefi.tefi(template, search, tefi_pixels, tefi_scales,
                  tefi_angles, thresh=-1.1, use_percentile=False, alpha=math.pi/2)

@pytest.mark.filterwarnings('ignore::UserWarning')  # skimage deprecation warnings to move to new defaults
def test_tefi(template, search):
    tefi_pixels = [(10, 10)]
    tefi_scales = np.ones(search.shape, dtype=float)
    tefi_angles = [3.14159265]

    pixel = ciratefi.tefi(template, search, tefi_pixels, tefi_scales, tefi_angles,
                                   thresh=tefi_thresh, use_percentile=True, alpha=math.pi/2,
                                   upsampling=10)

    assert np.equal((11.5, 11.5), (pixel[1], pixel[0])).all()

@pytest.mark.filterwarnings('ignore::UserWarning')  # skimage deprecation warnings to move to new defaults
@pytest.mark.parametrize("cifi_thresh, rafi_thresh, tefi_thresh, alpha, radii",[
                         (90,90,100,math.pi/2,list(range(1, 3)))])
def test_ciratefi(template, search, cifi_thresh, rafi_thresh, tefi_thresh, alpha, radii):
    results = ciratefi.ciratefi(template, search, upsampling=10, cifi_thresh=cifi_thresh,
                                rafi_thresh=rafi_thresh, tefi_thresh=tefi_thresh,
                                use_percentile=True, alpha=alpha, radii=radii)

    assert len(results) == 3
    assert (np.array(results[1], results[0]) < 1).all()
