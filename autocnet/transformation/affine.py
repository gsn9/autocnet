import logging
import time
import numpy as np
from plio.io.io_gdal import GeoDataset
from skimage import transform as tf

from autocnet.spatial import isis

log = logging.getLogger(__name__)

def estimate_affine_transformation(reference_image,
                                   moving_image,
                                   bcenter_x,
                                   bcenter_y,
                                   size_x=60,
                                   size_y=60):
    """
    Using the a priori sensor model, project corner and center points from the reference_image into
    the moving_image and use these points to estimate an affine transformation.

    Parameters
    ----------
    reference_image:  plio.io.io_gdal.GeoDataset
                source image
    moving_image: plio.io.io_gdal.GeoDataset
                destination image; gets matched to the source image
    bcenter_x:  int
                sample location of source measure in reference_image
    bcenter_y:  int
                line location of source measure in reference_image
    size_x:     int
                half-height of the subimage used in the affine transformation
    size_y:     int
                half-width of the subimage used in affine transformation

    Returns
    -------
    affine : object
             The affine transformation object

    """
    t1 = time.time()
    if not isinstance(moving_image, GeoDataset):
        raise Exception(f"Input cube must be a geodataset obj, but is type {type(moving_image)}.")
    if not isinstance(reference_image, GeoDataset):
        raise Exception(f"Match cube must be a geodataset obj, but is type {type(reference_image)}.")

    base_startx = int(bcenter_x - size_x)
    base_starty = int(bcenter_y - size_y)
    base_stopx = int(bcenter_x + size_x)
    base_stopy = int(bcenter_y + size_y)

    match_size = reference_image.raster_size

    # for now, require the entire window resides inside both cubes.
    if base_stopx > match_size[0]:
        raise Exception(f"Window: {base_stopx} > {match_size[0]}, center: {bcenter_x},{bcenter_y}")
    if base_startx < 0:
        raise Exception(f"Window: {base_startx} < 0, center: {bcenter_x},{bcenter_y}")
    if base_stopy > match_size[1]:
        raise Exception(f"Window: {base_stopy} > {match_size[1]}, center: {bcenter_x},{bcenter_y} ")
    if base_starty < 0:
        raise Exception(f"Window: {base_starty} < 0, center: {bcenter_x},{bcenter_y}")

    x_coords = [base_startx, base_startx, base_stopx, base_stopx, bcenter_x]
    y_coords = [base_starty, base_stopy, base_stopy, base_starty, bcenter_y]

    # Dispatch to the sensor to get the a priori pixel location in the input image
    lons, lats = isis.image_to_ground(reference_image.file_name, x_coords, y_coords, allowoutside=True)
    xs, ys = isis.ground_to_image(moving_image.file_name, lons, lats, allowoutside=True)

    # Check for any coords that do not project between images
    base_gcps = []
    dst_gcps = []
    for i, (base_x, base_y) in enumerate(zip(x_coords, y_coords)):
        if xs[i] is not None and ys[i] is not None:
            dst_gcps.append((xs[i], ys[i]))
            base_gcps.append((base_x, base_y))

    log.debug(f'base_gcps: {base_gcps}')
    log.debug(f'dst_gcps: {dst_gcps}')

    if len(dst_gcps) < 3:
        raise ValueError(f'Unable to find enough points to compute an affine transformation. Found {len(dst_corners)} points, but need at least 3.')

    affine = tf.estimate_transform('affine', np.array([*base_gcps]), np.array([*dst_gcps]))
    t2 = time.time()
    print(f'Estimation of the transformation took {t2-t1} seconds.')
    return affine