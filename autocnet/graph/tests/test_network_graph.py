import os
import pytest
import sys

import pandas as pd
from plio.io.io_controlnetwork import IsisControlNetwork

from autocnet.io.db import model
from autocnet.graph.network import NetworkCandidateGraph

from unittest.mock import patch, PropertyMock, MagicMock

if sys.platform.startswith("darwin"):
    pytest.skip("skipping DB tests for MacOS", allow_module_level=True)

@pytest.fixture()
def cnet():
    return IsisControlNetwork.from_dict({
            'id' : [1, 2, 3],
            'pointType' : [2]*3,
            'pointChoosername' : ['findfeatures']*3,
            'pointDatetime' : ['YYYY-MM-DDT00:00:00']*3,
            'pointEditLock': [False]*3,
            'pointIgnore' : [False]*3,
            'pointJigsawRejected': [False]*3,
            'referenceIndex' : [0]*3,
            'aprioriSurfPointSource': ['ground']*3,
            'aprioriSurfPointSourceFile' : ['ground.file']*3,
            'aprioriRadiusSource' : ['radius']*3, 
            'aprioriRadiusSourceFile' : ['radius.file']*3, 
            'latitudeConstrained' : [False]*3,
            'longitudeConstrained' : [False]*3, 
            'radiusConstrained' : [False]*3,
            'aprioriX' : [1017046.81161667, -1402345.22133465, 103571.17894436],
            'aprioriY' : [1017046.81161667, -1402345.22133465, 103571.17894436],
            'aprioriZ' : [1014022.55349016, -1404707.80219809, 101009.09763132],
            'aprioriCovar' : [[]]*3,
            'adjustedX' : [0]*3,
            'adjustedY' : [0]*3,
            'adjustedZ' : [0]*3,
            'adjustedCovar' : [[]]*3,
            'pointLog' : [[]]*3,
            'serialnumber' : ['SN1345', 'SN2348', 'SN9730'],
            'measureType' : [1]*3,
            'sample' : [2]*3,
            'line' : [1]*3,
            'sampleResidual' : [0.1]*3,
            'lineResidual' : [0.1]*3,
            'measureChoosername' : ['pointreg']*3,
            'measureDatetime' : ['YYYY-MM-DDT00:00:00']*3,
            'measureEditLock' : [False]*3,
            'measureIgnore': [False]*3,
            'measureJigsawRejected': [False]*3,
            'diameter' : [1000]*3,
            'apriorisample' : [0]*3,
            'aprioriline' : [0]*3,
            'samplesigma': [0]*3,
            'linesigma' : [0]*3,
            'measureLog' : [[]]*3
            }) 


"""@pytest.mark.parametrize("image_data, expected_npoints", [({'id':1, 'serial': 'BRUH'}, 1)])
def test_place_points_from_cnet(cnet, image_data, expected_npoints, ncg):
    with ncg.session_scope() as session:
        model.Images.create(session, **image_data)

        ncg.place_points_from_cnet(cnet)

        resp = session.query(model.Points)
        assert len(resp.all()) == expected_npoints
        assert len(resp.all()) == cnet.shape[0]"""

def test_to_isis(db_controlnetwork, ncg, node_a, node_b, tmpdir):
    ncg.add_edge(0,1)
    ncg.nodes[0]['data'] = node_a
    ncg.nodes[1]['data'] = node_b

    outpath = tmpdir.join('outnet.net')
    ncg.to_isis(outpath)

    assert os.path.exists(outpath)


def test_from_filelist(default_configuration, tmp_path, ncg):
    # Written as a list and not parametrized so that the fixture does not automatically clean
    #  up the DB. Needed to test the functionality of the clear_db kwarg.
    for filelist, clear_db in [(['bar1.cub', 'bar2.cub', 'bar3.cub'], False),
                               ([], True),
                               (['bar1.cub', 'bar2.cub', 'bar3.cub'], True)]:
        filelist = [tmp_path/f for f in filelist]

        # Since we have no overlaps (everything is faked), len(ncg) == 0
        test_ncg = NetworkCandidateGraph.from_filelist(filelist, default_configuration, clear_db=clear_db)
        
        with test_ncg.session_scope() as session:
            res = session.query(model.Images).all()
            assert len(res) == len(filelist)
    

def test_global_clear_db(ncg):
    i = model.Images(name='foo', path='/fooland/foo.img')
    with ncg.session_scope() as session:
        session.add(i)

        res = session.query(model.Images).all()
        assert len(res) == 1

    ncg.clear_db()

    with ncg.session_scope() as session:
        res = session.query(model.Images).all()
        assert len(res) == 0

def test_selective_clear_db(ncg):
    i = model.Images(name='foo', path='fooland/foo.img')
    p = model.Points(pointtype=2)

    with ncg.session_scope() as session:
        session.add(i)
        session.add(p)

        res = session.query(model.Images).all()
        assert len(res) == 1
        res =  session.query(model.Points).all()
        assert len(res) == 1
    
    ncg.clear_db(tables=['Points'])

    with ncg.session_scope() as session:
        res = session.query(model.Images).all()
        assert len(res) == 1
        res = session.query(model.Points).all()
        assert len(res) == 0

def test_cnet_to_db(ncg, cnet):
    # check that the resulting DB DFs have same columns as corresponding Models
    imgs = [model.Images(name='foo1', serial=cnet.iloc[0]['serialnumber']),
            model.Images(name='foo2', serial=cnet.iloc[1]['serialnumber']),
            model.Images(name='foo3', serial=cnet.iloc[2]['serialnumber'])]

    with ncg.session_scope() as session:
        session.add_all(imgs)

    p_df, m_df = ncg.cnet_to_db(cnet)
    
    point_columns = model.Points.__table__.columns.keys()
    measure_columns = model.Measures.__table__.columns.keys()
    
    for key in point_columns:
        assert key in p_df.columns, f"column \'{key}\' not in points dataframe"
    for key in measure_columns:
        assert key in m_df.columns, f"column \'{key}\' not in measures dataframe"

# TO DO: test the clear tables functionality on ncg.place_points_from_cnet
