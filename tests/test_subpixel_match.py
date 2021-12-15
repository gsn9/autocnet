import pytest
from autocnet.io.db.model import Points, Measures, Images, Overlay
from autocnet.matcher.subpixel import smart_register_point

@pytest.mark.long
def test_full_subpixel_registration(ncg, images, point):
    # This is a full integration test for subpixel matching.

    print(__file__)

    with ncg.session_scope() as session:
        for image in images:
            Images.create(session, **image)

        # Create the overlap
        Overlay.create(session, **{'id': 1, 'intersections':[8325, 17517, 90885]})
        # Create doesn't work nicely with a full dict, because it 
        measures = point['measures']
        point['measures'] = []
        Points.create(session, **point)
        for measure in measures:
            Measures.create(session, **measure)

        # Confirm inserts
        res = session.query(Images).all()
        assert len(res) == 3
        res = session.query(Points).one()
        assert isinstance(res, Points)
        assert len(res.measures) == 3

        session.flush()

        shared_kwargs = {'geom_func':'simple',
                 'match_func':'classic',
                 'cost_func':lambda x,y:y,
                 'chooser':'smart_subpixel_registration'}

        parameters = [
            {'match_kwargs': {'image_size':(121,121), 'template_size':(61,61)}},
            {'match_kwargs': {'image_size':(151,151), 'template_size':(67,67)}},
            {'match_kwargs': {'image_size':(181,181), 'template_size':(73,73)}},
            {'match_kwargs': {'image_size':(221,221), 'template_size':(81,81)}},
            {'match_kwargs': {'image_size':(251,251), 'template_size':(89,89)}},
            {'match_kwargs': {'image_size':(281,281), 'template_size':(98,98)}}
        ]

        measures_to_update, measures_to_set_false = smart_register_point(1, 
                             parameters=parameters,
                             shared_kwargs=shared_kwargs,
                             ncg=ncg)

        print(measures_to_update)

        for measure in measures_to_update:
            if measure['_id'] == 2:
                assert measure['line'] == 271.00253223520383
                assert measure['sample'] == 266.9152655856491
                assert measure['ignore'] == False
            elif measure['_id'] == 3:
                assert measure['line'] == 258.6752555930361
                assert measure['sample'] == 258.80172907976646
                assert measure['ignore'] == False
        
    with ncg.session_scope() as session:
        m1 = session.query(Measures).filter(Measures.id == 2).one()
        
        assert m1.line == pytest.approx(271.00253223520383, 6)
        assert m1.sample == pytest.approx(266.9152655856491, 6)
        assert m1.ignore == False

        m2 = session.query(Measures).filter(Measures.id == 3).one()
        assert m2.line == pytest.approx(258.6752555930361, 6)
        assert m2.sample == pytest.approx(258.80172907976646, 6)
        assert m2.ignore == False
