import numpy as np
import pandas as pd
import pytest

import os
import sys

sys.path.insert(0, '..')

from autocnet.graph.network import CandidateGraph

@pytest.fixture(scope='module')
def candidategraph():
    a = '/fake_path/a.img'
    b = '/fake_path/b.img'
    c = '/fake_path/c.img'
    adj = {a:[b,c],
           b:[a,c],
           c:[a, b]}
    cg = CandidateGraph.from_adjacency(adj)

    match_data = np.array([[0.0,	188.0,	1.0, 0.0, 170.754211],
                           [0.0,	189.0,	1.0, 0.0, 217.451141],
                           [0.0,	185.0,	1.0, 1.0, 108.843925]])
    matches = pd.DataFrame(match_data, columns=['source_image', 'source_idx',
                                                'destination_image', 'destination_idx',
                                                'distance'])
    masks = pd.DataFrame([[True, True],
                          [False, True],
                          [True, False]],
                          columns=['rain', 'maker'])

    for s, d, e in cg.edges.data('data'):
        e['fundamental_matrix'] = np.random.random(size=(3,3))
        e.matches = matches
        e.masks = masks
        e['source_mbr'] = [[0,1], [0,2]]
        e['destin_mbr'] = [[0.5, 0.5], [1, 1]]

    kps = np.array([[233.358475,	105.288162,	0.035672, 4.486887,	164.181046,	0.0, 1.0],
                    [366.288116,	98.761131,	0.035900, 4.158592,	147.278580,	0.0, 1.0],
                    [170.932114,	114.173912,	0.037852, 94.446655, 0.401794,	0.0, 3.0]])

    keypoints = pd.DataFrame(kps, columns=['x', 'y', 'response', 'size', 'angle',
                                           'octave', 'layer'])

    for i, n in cg.nodes.data('data'):
        n.keypoints = keypoints
        n.descriptors = np.random.random(size=(3, 128))
        n.masks = masks

    return cg


@pytest.fixture()
def images():
    images = [
      {
        "id": 8325,
        "name": "B08_012650_1780_XN_02S046W",
        "path": 'tests/test_subpixel_match/B08_012650_1780_XN_02S046W.l1.cal.destriped.crop.cub',
        "serial": "MRO/CTX/0923633304:171",
        "cam_type": None,
        "geom": None
      },
      {
        "id": 17517,
        "name": "J04_046447_1777_XI_02S046W",
        "path": 'tests/test_subpixel_match/J04_046447_1777_XI_02S046W.l1.cal.destriped.crop.cub',
        "serial": "MRO/CTX/1151163058:226",
        "geom": None,
        "cam_type": None
      },
      {
        "id": 25886,
        "name": "D16_033458_1785_XN_01S046W",
        "path": 'tests/test_subpixel_match/D16_033458_1785_XN_01S046W.l1.cal.destriped.crop.cub',
        "serial": "MRO/CTX/1063720721:242",
        "geom": None,
        "cam_type": None
      }
    ]
    return images

@pytest.fixture
def point():
    point = {
        "id": 1,
        "identifier": "ppio_ns9_ew7",
        "overlapid": 1,
        "cam_type": "isis",
        #"adjusted": "POINT Z (2325037.0927991 -2469370.8555417 -134565.97138111)",
        "measures": [
        {
            "id": 1,
            "pointid": 1,
            "imageid": 8325,
            "sample": 250.7802039064, #3811.7802039064,
            "line": 250.4993337409, #1120.4993337409,
            "apriorisample": 250.7802039064, #3811.7802039064,  # 3561 start
            "aprioriline":  250.4993337409, #1120.4993337409,  # 870 start
            "sampler": None,
            "liner": None,
            "residual": None,
            "samplesigma": None,
            "linesigma": None,
            "weight": None,
            "rms": None,
            "ignore": False,
            "jigreject": False,
            "template_shift": 0,
            "template_metric": 1,
            "phase_error": 0,
            "phase_shift": 0,
            "serial": "MRO/CTX/0923633304:171",
            "measuretype": 3,
            "phase_diff": 0,
            "choosername": "place_points_in_overlap"
        },
        {
            "id": 2,
            "pointid": 1,
            "imageid": 17517,
            "sample": 250.9395403013, #3393.86972928868,
            "line": 250.839748464, #26526.0063355324,
            "apriorisample": 250.9395403013, #3377.9395403013,
            "aprioriline": 250.839748464, #26505.839748464,
            "sampler": None,
            "liner": None,
            "residual": None,
            "samplesigma": None,
            "linesigma": None,
            "weight": -0.675201815708463,
            "rms": None,
            "ignore": False,
            "jigreject": False,
            "template_shift": 26.2969817659746,
            "template_metric": 0.586938142776489,
            "phase_error": None,
            "phase_shift": None,
            "serial": "MRO/CTX/1151163058:226",
            "measuretype": 3,
            "phase_diff": None,
            "choosername": "smart_subpixel_registration"
        },{
        "id": 3,
		"pointid": 1,
		"imageid": 25886,
		"sample": 244.07172109498, #259.695805803292,
		"line": 250.5145259831, #2266.63178109482,
		"apriorisample": 244.07172109498, #244.07172109498,
		"aprioriline": 250.5145259831, #2258.5145259831,
		"sampler": None,
		"liner": None,
		"residual": None,
		"samplesigma": None,
		"linesigma": None,
		"weight": -0.629346802575788,
		"rms": None,
		"ignore": False,
		"jigreject": False,
		"template_shift": 20.1108257538073,
		"template_metric": 0.558610916137695,
		"phase_error": None,
		"phase_shift": None,
		"serial": "MRO/CTX/1063720721:242",
		"measuretype": 3,
		"phase_diff": None,
		"choosername": "smart_subpixel_registration"
	    }
        ],
        "ignore": False,
        "pointtype": 2
    }
    return point