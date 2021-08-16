import os
import warnings

import numpy as np
import pandas as pd
from plio.io.io_gdal import GeoDataset
import pvl
from shapely.geometry import Point
from geoalchemy2.functions import ST_DWithin

from autocnet.io.db.model import Points, Measures, Images, CandidateGroundPoints
from autocnet.spatial.isis import isis2np_types
from autocnet.graph.node import NetworkNode
from autocnet.matcher.subpixel import check_geom_func, check_match_func, geom_match_simple
from autocnet.matcher.cpu_extractor import extract_most_interesting

from autocnet.utils.utils import bytescale

from autocnet.spatial import isis
from autocnet.transformation.spatial import reproject, oc2og
from autocnet.io.db.model import Images
from autocnet.transformation import roi

def propagate_ground_point(point,
                           match_func='classic',
                           verbose=False,
                           match_kwargs={'image_size': (39, 39), 'template_size': (21, 21)},
                           cost=lambda x, y: y == np.max(x),
                           threshold=0.01,
                           ncg=None,
                           preprocess=None,
                           Session=None):
    print(f'Attempting to propagate point {point.id}.')

    match_func = check_match_func(match_func)

    with ncg.session_scope() as session:
        query = session.query(Images).filter(Images.geom.ST_Intersects(point._geom))
        images = pd.read_sql(query.statement, ncg.engine)

    path = point.path
    sy = point.line
    sx = point.sample
    pointid = point.id

    base_image = GeoDataset(path)

    lon = point.geom.x
    lat = point.geom.y

    # list of matching results in the format:
    # [measure_index, x_offset, y_offset, offset_magnitude]
    best_correlation = -np.inf
    best_match = None
    for _, image in images.iterrows():
        # When grounding to THEMIS the df has a PATH to the QUAD
        dest_image = GeoDataset(image["path"])
        #if os.path.basename(m['path']) == os.path.basename(image['path']):
        #    continue
        try:
            print(f'prop point: base_image: {base_image}')
            print(f'prop point: dest_image: {dest_image}')
            print(f'prop point: (sx, sy): ({sx}, {sy})')
            x,y, dist, metrics, corrmap = geom_match_simple(base_image, dest_image, sx, sy, 25, 25, \
                    match_func = match_func, \
                    match_kwargs=match_kwargs, \
                    preprocess=preprocess,
                    verbose=verbose)
        except Exception as e:
            print(e)
            continue
        
        if x is None:
            print(f'Match returned None. Unable to match the ground point into image {image["path"]}.')
            continue
        
        current_cost = cost(dist, metrics)
        if current_cost > best_correlation and current_cost >= threshold:
            print(f'Found a match with correlation: {metrics}, shift distance: {dist}, and cost {current_cost} ', )
            best_correlation = current_cost
            best_match = [x, y, metrics, dist, corrmap, path, image['path'], image['id'], image['serial']]
        else:
            print(f'Found a match, but that match is either less than the threshold or worse than the current best.')
            print(f'Found a match with correlation: {metrics}, shift distance: {dist}, and cost {current_cost} ', )
    
    if best_match is None:
        print('Unable to propagate this ground point into any images.')
        return
    
    dem = ncg.dem
    config = ncg.config

    height = dem.get_height(lat, lon)

    semi_major = config['spatial']['semimajor_rad']
    semi_minor = config['spatial']['semiminor_rad']
    # The CSM conversion makes the LLA/ECEF conversion explicit
    # reprojection takes ographic lat
    lon_oc = lon
    lat_oc = lat

    lon_og, lat_og = oc2og(lon_oc, lat_oc, semi_major, semi_minor)
    x, y, z = reproject([lon_og, lat_og, height],
                         semi_major, semi_minor,
                         'latlon', 'geocent')

    row = best_match
    sample  = float(row[0])
    line = float(row[1])

    # Create the point
    # Question - do we need to get the z from the ground source?
    point_geom = Point(x,y,z)
    cam_type = 'isis'
    point = Points(apriori=point_geom,
                   adjusted=point_geom,
                   pointtype=3, # Would be 3 or 4 for ground
                   cam_type=cam_type,
                   reference_index=0)

    # Add the measure that was the best match.
    # Set the line/sample and aprioriline/apriorisample to be identical.
    point.measures.append(Measures(sample=sample,
                                   line=line,
                                   apriorisample=sample,
                                   aprioriline=line,
                                   imageid=row[7],
                                   serial=row[8],
                                   measuretype=2,
                                   weight=float(row[2]),  # metric
                                   choosername='propagate_ground_point'))

    for _, image in images.iterrows():
        if cam_type == "csm":
            # BROKEN
            image_coord = node.camera.groundToImage(gnd)
            sample, line = image_coord.samp, image_coord.line
        if cam_type == "isis":
            try:
                sample, line = isis.ground_to_image(image["path"], lon_oc, lat_oc)
            except ValueError as e:
                if 'Requested position does not project in camera model' in e.stderr:
                    print(f'interesting point ({lon_oc},{lat_oc}) does not project to image {images["image_path"]}')
                    continue

        if image['id'] == best_match[7]:
            continue  # The measures was already added above, simply update the apriori line/sample

        point.measures.append(Measures(sample=float(sample),
                                        line=float(line),
                                        apriorisample=float(sample),
                                        aprioriline=float(line),
                                        imageid=image['id'],
                                        serial=image['serial'],
                                        measuretype=3,
                                        choosername='propagate_ground_point'))
    print(f'Adding {len(point.measures)} measures.')
    with ncg.session_scope() as session:
        session.add(point)
    print('Done adding points')

def find_most_interesting_ground(apriori_lon_lat,
                                 ground_mosaic,
                                 cam_type='isis',
                                 size=71, 
                                 threshold=0.01,
                                 ncg=None,
                                 Session=None):
    """
    This is the same functionality as cim.generate_ground_points. The difference here
    is that the data are pushed to a database table instead of being pushed to
    a
    Parameters
    ----------
    cam_type : str
               Either 'isis' (Default;enabled) or 'csm' (Disabled). Defines which sensor model implementation to use.
    size : int
           The size of the area to extract from the data to search for interesting features.
    base_dtype : str
                 The numpy string that describes the datatype of the base image. Options include 'int8', 'uint8', 'float32'.
    """
    if cam_type == 'csm':
        raise ValueError('Unable to find interesting ground using a CSM sensor.')

    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    if not isinstance(ground_mosaic, GeoDataset):
        ground_mosaic = GeoDataset(ground_mosaic)

    p = Point(*apriori_lon_lat)

    # Convert the apriori lon, lat into line,sample in the image
    linessamples = isis.point_info(ground_mosaic.file_name, p.x, p.y, 'ground')
    line = linessamples.get('Line')
    sample = linessamples.get('Sample')

    try:
        base_dtype = isis2np_types[pvl.load(ground_mosaic.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    except:
        s_image_dtype = None

    # Get the most interesting feature in the area
    image = roi.Roi(ground_mosaic, sample, line, size_x=size, size_y=size)
    image_roi = image.clip(dtype=base_dtype)

    interesting = extract_most_interesting(image_roi,  extractor_parameters={'nfeatures':30})

    if interesting is None:
        warnings.warn('No interesting feature found. This is likely caused by either large contiguous no data areas in the base or a mismatch in the base_dtype.')
        return

    left_x, _, top_y, _ = image.image_extent
    newsample = left_x + interesting.x
    newline = top_y + interesting.y

    # @LAK - this needs eyeballs to confirm correct oc/og
    newpoint = isis.point_info(ground_mosaic.file_name, newsample, newline, 'image')
    p = Point(newpoint.get('PositiveEast360Longitude'),
              newpoint.get('PlanetocentricLatitude'))

    with ncg.session_scope() as session:
        # Check to see if the point already exists
        g = CandidateGroundPoints(path=ground_mosaic.file_name,
                    choosername='find_most_interesting_ground',
                    aprioriline=line,
                    apriorisample=sample,
                    line=newline,
                    sample=newsample,
                    geom=p,
                    ignore=False)

        res = session.query(CandidateGroundPoints).filter(ST_DWithin(CandidateGroundPoints._geom, g._geom, threshold)).all()
        if res:
            warnings.warn(f'Skipping adding a point as another point already exists within {threshold} units.')
        else:
            session.add(g)

def find_ground_reference(point,
                           ncg=None,
                           Session=None,
                           geom_func='simple',
                           match_func='classic',
                           match_kwargs={},
                           geom_kwargs={"size_x": 16, "size_y": 16},
                           threshold=0.9,
                           cost_func=lambda x,y: (0*x)+y,
                           verbose=False):

    geom_func = check_geom_func(geom_func)
    match_func = check_match_func(match_func)

    # Get the roi to match from the base image
    with ncg.session_scope() as session:
        measures = session.query(Measures).filter(Measures.pointid == point.id).all()

        for m in measures:
            if m.measuretype == 0:
                base = m
                bsample = base.sample
                bline = base.line
        baseimage = base.serial # We are piggybacking the base image name onto the measure serial.
    if not os.path.exists(baseimage):
        raise FileNotFoundError(f'Unable to find {baseimage} to register the data to.')

    # Get the base image and the roi extracted that the image data will register to
    baseimage = GeoDataset(baseimage)

    # Select the images that the point is in.
    cost = -1
    sample = None
    line = None
    best_node = None

    with ncg.session_scope() as session:
        images = session.query(Images).filter(Images.geom.ST_Intersects(point._geom)).all()

        nodes = []
        for image in images:
            node = NetworkNode(node_id=image.id, image_path=image.path)
            nodes.append(node)

    for node in nodes:
        node.geodata
        image_geodata = node.geodata

        x, y, dist, metrics, _ = geom_func(baseimage, image_geodata,
                                            bsample, bline,
                                            match_func = match_func,
                                            match_kwargs = match_kwargs,
                                            verbose=verbose,
                                            **geom_kwargs)
        if x == None:
            print(f'Unable to match image {node["image_name"]} to {baseimage}.')
            continue

        current_cost = cost_func(dist, metrics)
        print(f'Results returned: {current_cost}.')
        if current_cost >= cost and current_cost >= threshold:
            cost = current_cost
            sample = x
            line = y
            best_node = node
        else:
            print(f'Cost function not met. Unable to use {node["image_name"]} as reference')
    if sample == None:
        print('Unable to register this point to a ground source.')
        return

    # A reference measure has been identified. This measure matched successfully to the ground.
    # Get the lat/lon from the sample/line
    reference_node = best_node
    print('Success...')
    # Setup the measures

    m = Measures(sample=sample,
                line=line,
                apriorisample=sample,
                aprioriline=line,
                imageid=node['node_id'],
                serial=node.isis_serial,
                measuretype=3,
                choosername='add_measures_to_ground')

    with ncg.session_scope() as session:
        point = session.query(Points).filter(Points.id == point.id).one()

        point.measures.append(m)
        point.reference_index = len(point.measures) - 1  # The measure that was just appended is the new reference

    print('successfully added a reference measure to the database.')
