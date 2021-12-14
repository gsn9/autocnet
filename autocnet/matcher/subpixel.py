from collections import defaultdict
import json
from math import modf, floor
import time
import numpy as np
import warnings

from subprocess import CalledProcessError

import numbers

import sys

import cv2

from skimage import transform as tf
from skimage import registration 
from skimage import filters
from skimage.util import img_as_float32
from scipy import fftpack

from matplotlib import pyplot as plt

from plio.io.io_gdal import GeoDataset

import pvl

import PIL
from PIL import Image

from autocnet.matcher.naive_template import pattern_match, pattern_match_autoreg
from autocnet.matcher.mutual_information import mutual_information_match
from autocnet.matcher import ciratefi
from autocnet.matcher.mutual_information import mutual_information
from autocnet.spatial import isis 
from autocnet.io.db.model import Measures, Points, Images, JsonEncoder
from autocnet.graph.node import NetworkNode
from autocnet.transformation import roi
from autocnet.transformation.affine import estimate_affine_transformation
from autocnet import spatial
from autocnet.utils.utils import bytescale

from sqlalchemy import inspect

PIL.Image.MAX_IMAGE_PIXELS = sys.float_info.max

def check_geom_func(func):
    # TODO: Pain. Stick with one of these and delete this function along with
    # everything else
    geom_funcs = {
            "classic": geom_match_classic,
            "new": geom_match,
            "simple" : geom_match_simple,
    }

    if func in geom_funcs.values():
        return func

    if func in geom_funcs.keys():
        return geom_funcs[func]

    raise Exception(f"{func} not a valid geometry function.")

def check_match_func(func):
    match_funcs = {
        "classic": subpixel_template_classic,
        "phase": iterative_phase,
        "template": subpixel_template,
        "mutualinformation": mutual_information_match
    }

    if func in match_funcs.values():
        return func

    if func in match_funcs.keys():
        return match_funcs[func]

    raise Exception(f"{func} not a valid matching function.")

# TODO: look into KeyPoint.size and perhaps use to determine an appropriately-sized search/template.
def _prep_subpixel(nmatches, nstrengths=2):
    """
    Setup the data strutures to return for subpixel matching.

    Parameters
    ----------
    nmatches : int
                The number of pixels to be subpixel matches

    nstrengths : int
                    The number of 'strength' values to be returned
                    by the subpixel matching method.

    Returns
    -------
    shifts_x : ndarray
               (nmatches, 1) to store the x_shift parameter

    shifts_y : ndarray
               (nmatches, 1) to store the y_shift parameter

    strengths : ndarray
                (nmatches, nstrengths) to store the strengths for each point

    new_x : ndarray
            (nmatches, 1) to store the updated x coordinates

    new_y : ndarray
            (nmatches, 1) to store the updated y coordinates
    """
    # Setup to store output to append to dataframes
    shifts_x = np.zeros(nmatches)
    shifts_y = np.zeros(nmatches)
    strengths = np.zeros((nmatches, nstrengths))

    new_x = np.empty(nmatches)
    new_y = np.empty(nmatches)

    return shifts_x, shifts_y, strengths, new_x, new_y

def check_image_size(imagesize):
    """
    Given an x,y tuple, ensure that the values
    are odd. Used by the subpixel template to also ensure
    that the template size is the one requested and not 2x
    the template size.

    Parameters
    ----------
    imagesize : tuple
                in the form (size_x, size_y)
    """
    if isinstance(imagesize, numbers.Number):
        imagesize = (int(imagesize), int(imagesize))


    x = imagesize[0]
    y = imagesize[1]

    if x % 2 == 0:
        x += 1
    if y % 2 == 0:
        y += 1
    x = floor(x/2)
    y = floor(y/2)
    return x,y

def clip_roi(img, center_x, center_y, size_x=200, size_y=200, dtype="uint64"):
    """
    Given an input image, clip a square region of interest
    centered on some pixel at some size.
    Parameters
    ----------
    img : ndarray or object
          The input image to be clipped or an object
          with a read_array method that takes a pixels
          argument in the form [xstart, ystart, xstop, ystop]
    center_x : Numeric
               The x coordinate to the center of the roi
    center_y : Numeric
               The y coordinate to the center of the roi
    img_size : int
               1/2 of the total image size. This value is the
               number of pixels grabbed from each side of the center
    Returns
    -------
    clipped_img : ndarray
                  The clipped image
    """

    try:
        raster_size = img.raster_size
    except:
        # x,y form
        raster_size = img.shape[::-1]
    axr, ax = modf(center_x)
    ayr, ay = modf(center_y)

    if ax + size_x > raster_size[0]:
        size_x = floor(raster_size[0] - center_x)
    if ax - size_x < 0:
        size_x = int(ax)
    if ay + size_y > raster_size[1]:
        size_y =floor(raster_size[1] - center_y)
    if ay - size_y < 0:
        size_y = int(ay)

    # Read from the upper left origin
    pixels = [ax-size_x, ay-size_y, size_x*2, size_y*2]
    pixels = list(map(int, pixels))  #
    if isinstance(img, np.ndarray):
        subarray = img[pixels[1]:pixels[1] + pixels[3] + 1, pixels[0]:pixels[0] + pixels[2] + 1]
    else:
        try:
            subarray = img.read_array(pixels=pixels, dtype=dtype)
        except:
            return None, 0, 0
    return subarray, axr, ayr

def subpixel_phase(sx, sy, dx, dy,
                   s_img, d_img,
                   image_size=(51, 51),
                   **kwargs):
    """
    Apply the spectral domain matcher to a search and template image. To
    shift the images, the x_shift and y_shift, need to be subtracted from
    the center of the search image. It may also be necessary to apply the
    fractional pixel adjustment as well (if for example the center of the
    search is not an integer); this function do not manage shifting.

    Parameters
    ----------
    template : ndarray
               The template used to search

    search : ndarray
             The search image

    Returns
    -------
    x_offset : float
               Shift in the x-dimension

    y_offset : float
               Shift in the y-dimension

    strength : tuple
               With the RMSE error and absolute difference in phase
    """
    image_size = check_image_size(image_size)

    s_roi = roi.Roi(s_img, sx, sy, size_x=image_size[0], size_y=image_size[1])
    d_roi = roi.Roi(d_img, dx, dy, size_x=image_size[0], size_y=image_size[1])

    s_image = s_roi.array
    d_template = d_roi.array

    if s_image.shape != d_template.shape:

        s_size = s_image.shape
        d_size = d_template.shape
        updated_size_x = int(min(s_size[1], d_size[1]))
        updated_size_y = int(min(s_size[0], d_size[0]))

        # Have to subtract 1 from even entries or else the round up that
        # occurs when the size is split over the midpoint causes the
        # size to be too large by 1.
        if updated_size_x % 2 == 0:
            updated_size_x -= 1
        if updated_size_y % 2 == 0:
            updated_size_y -= 1

        # Since the image is smaller than the requested size, set the size to
        # the current maximum image size and reduce from there on potential
        # future iterations.
        size = check_image_size((updated_size_x, updated_size_y))
        s_roi = roi.Roi(s_img, sx, sy,
                        size_x=size[0], size_y=size[1])
        d_roi = roi.Roi(d_img, dx, dy,
                        size_x=size[0], size_y=size[1])
        s_image = s_roi.array
        d_template = d_roi.array

        if (s_image is None) or (d_template is None):
            return None, None, None

    (shift_y, shift_x), error, diffphase = registration.phase_cross_correlation(s_image, d_template, **kwargs)
    dx = d_roi.x - shift_x
    dy = d_roi.y - shift_y

    return dx, dy, error, None

def subpixel_transformed_template(sx, sy, dx, dy,
                                  s_img, d_img,
                                  transform,
                                  image_size=(251, 251),
                                  template_size=(51, 51),
                                  template_buffer=5,
                                  func=pattern_match,
                                  verbose=False,
                                  **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.

    Parameters
    ----------
    sx : Numeric
         Source X coordinate

    sy : Numeric
         Source y coordinate

    dx : Numeric
         The desintation x coordinate

    dy : Numeric
         The destination y coordinate

    s_img : GeoDataset
            The source image GeoDataset

    d_img : GeoDataset
            The destination image GeoDataset

    transform : object
                A skiage transform object that has scale. The transform object is
                used to project the template into the image.

    image_size : tuple
                 (xsize, ysize) of the image that is searched within (this should be larger
                 than the template size)

    template_size : tuple
                    (xsize, ysize) of the template to iterate over the image in order
                    to identify the area(s) of highest correlation.

    template_buffer : int
                      The inverse buffer applied to the transformed template image. When
                      the warp is applied to project from the template into the image, some
                      amount of no data exists around the edges. This variable is used to clip
                      some number of pixels off the edges of the template. The higher the rotation
                      the higher this value should be.

    func : callable
           The function used to pattern match

    verbose : bool
              If true, generate plots of the matches

    Returns
    -------
    x_shift : float
               Shift in the x-dimension

    y_shift : float
               Shift in the y-dimension

    strength : float
               Strength of the correspondence in the range [-1, 1]

    corrmap : ndarray
              An n,m array of correlation coefficients

    See Also
    --------
    autocnet.matcher.naive_template.pattern_match : for the kwargs that can be passed to the matcher
    autocnet.matcher.naive_template.pattern_match_autoreg : for the jwargs that can be passed to the autoreg style matcher
    """
    image_size = check_image_size(image_size)
    template_size = check_image_size(template_size)

    template_size_x = int(template_size[0] * transform.scale[0])
    template_size_y = int(template_size[1] * transform.scale[1])

    s_roi = roi.Roi(s_img, sx, sy, size_x=image_size[0], size_y=image_size[1])
    d_roi = roi.Roi(d_img, dx, dy, size_x=template_size_x, size_y=template_size_y)

    if not s_roi.is_valid or not d_roi.is_valid:
        return [None] * 4

    try:
        s_image_dtype = isis.isis2np_types[pvl.load(s_img.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    except:
        s_image_dtype = None

    try:
        d_template_dtype = isis.isis2np_types[pvl.load(d_img.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    except:
        d_template_dtype = None

    s_image = bytescale(s_roi.array)
    d_template = bytescale(d_roi.array)

    if verbose:
        fig, axs = plt.subplots(1, 5, figsize=(20,10))
        # Plot of the original image and template
        axs[0].imshow(s_image, cmap='Greys')
        axs[0].set_title('Destination')
        axs[1].imshow(d_template, cmap='Greys')
        axs[1].set_title('Original Source')

    # Build the transformation chance
    shift_x, shift_y = d_roi.center

    tf_rotate = tf.AffineTransform(rotation=transform.rotation, shear=transform.shear)
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

    # Define the full chain and the inverse
    trans = (tf_shift + (tf_rotate + tf_shift_inv))
    itrans = trans.inverse

    # Now apply the affine transformation
    transformed_roi = tf.warp(d_template,
                                itrans,
                                order=3)

    # Scale the source arr to the destination array size
    scale_y, scale_x = transform.scale
    template_shape_y, template_shape_x = d_template.shape
    scaled_roi = tf.resize(transformed_roi, (int(template_shape_x/scale_x), int(template_shape_y/scale_x)))

    # Clip the transformed template to avoid no data around around the edges
    buffered_template = scaled_roi[template_buffer:-template_buffer,template_buffer:-template_buffer]

    # Apply the matcher on the transformed array
    shift_x, shift_y, metrics, corrmap = func(bytescale(buffered_template), s_image, **kwargs)

    # Hard check here to see if we are on the absolute edge of the template
    max_coord = np.unravel_index(corrmap.argmax(), corrmap.shape)[::-1]
    if 0 in max_coord or corrmap.shape[0]-1 == max_coord[0] or corrmap.shape[1]-1 == max_coord[1]:
        warnings.warn('Maximum correlation is at the edge of the template. Results are ambiguous.', UserWarning)
        return [None] * 4

    if verbose:
        axs[2].imshow(transformed_roi, cmap='Greys')
        axs[2].set_title('Affine Transformed Source')
        axs[3].imshow(buffered_template, cmap='Greys')
        axs[3].set_title('Scaled and Buffered Source')
        axs[4].imshow(corrmap)
        axs[4].set_title('Correlation')
        plt.show()

    # Project the center into the affine space
    projected_center = itrans(d_roi.center)[0]

    # Shifts need to be scaled back into full resolution, affine space
    shift_x *= scale_x
    shift_y *= scale_y

    # Apply the shifts (computed using the warped image) to the affine space center
    new_projected_x = projected_center[0] - shift_x
    new_projected_y = projected_center[1] - shift_y

    # Project the updated location back into image space
    new_unprojected_x, new_unprojected_y = trans([new_projected_x, new_projected_y])[0]

    # Apply the shift
    dx = d_roi.x - (d_roi.center[0] - new_unprojected_x)
    dy = d_roi.y - (d_roi.center[1] - new_unprojected_y)

    return dx, dy, metrics, corrmap

def subpixel_template_classic(sx, sy, dx, dy,
                              s_img, d_img,
                              image_size=(251, 251),
                              template_size=(51,51),
                              func=pattern_match,
                              **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.
    Parameters
    ----------
    sx : Numeric
         Source X coordinate
    sy : Numeric
         Source y coordinate
    dx : Numeric
         The desintation x coordinate
    dy : Numeric
         The destination y coordinate
    s_img : GeoDataset
            The source image GeoDataset
    d_img : GeoDataset
            The destination image GeoDataset
    image_size : tuple
                 (xsize, ysize) of the image that is searched within (this should be larger
                 than the template size)
    template_size : tuple
                    (xsize, ysize) of the template to iterate over the image in order
                    to identify the area(s) of highest correlation.

    Returns
    -------
    x_shift : float
              Shift in the x-dimension
    y_shift : float
              Shift in the y-dimension
    strength : float
               Strength of the correspondence in the range [-1, 1]
    See Also
    --------
    autocnet.matcher.naive_template.pattern_match : for the kwargs that can be passed to the matcher
    autocnet.matcher.naive_template.pattern_match_autoreg : for the jwargs that can be passed to the autoreg style matcher
    """

    # Image or source is the reference that the template is registered to

    image_size = check_image_size(image_size)
    template_size = check_image_size(template_size)

    # In ISIS source image is the search and destination image is the pattern.
    # In ISIS the search is CTX and the pattern is THEMIS
    # So the data that are being used are swapped between autocnet and ISIS.
    s_roi = roi.Roi(s_img, sx, sy, size_x=image_size[0], size_y=image_size[1])
    d_roi = roi.Roi(d_img, dx, dy, size_x=template_size[0], size_y=template_size[1])

    """print('Source: ', sx, sy, d_roi.x, d_roi.y)
    print('Destination ',dx, dy, s_roi.x, s_roi.y )

    print('d shape', d_roi.clip().shape)
    print('d mean: ', d_roi.clip().mean())
    print(f'd mm: {d_roi.clip().min()} {d_roi.clip().max()}')"""
    #print(f'{len(isis.get_isis_special_pixels(d_roi.clip()))} chip sps : ', isis.get_isis_special_pixels(d_roi.clip()))

    s_image = s_roi.array
    d_template = d_roi.array

    """print('s shape', s_image.shape)
    print('s mean: ', s_image.mean())
    print(f's mm: {s_image.min()} {s_image.max()}')"""
    #print(f'{len(isis.get_isis_special_pixels(s_image))} chip sps: ', isis.get_isis_special_pixels(s_image))

    if d_roi.variance == 0:
        warnings.warn('Input ROI has no variance.')
        return [None] * 4

    if (s_image is None) or (d_template is None):
        return None, None, None, None

    shift_x, shift_y, metrics, corrmap = func(img_as_float32(d_template), img_as_float32(s_image), **kwargs)
    if shift_x is None:
        return None, None, None, None
    # Apply the shift and return
    dx = d_roi.x - shift_x
    dy = d_roi.y - shift_y
    return dx, dy, metrics, corrmap

def subpixel_template(sx, sy, dx, dy,
                      s_img, d_img,
                      image_size=(251, 251),
                      template_size=(51,51),
                      func=pattern_match,
                      verbose=False,
                      **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.

    Parameters
    ----------
    sx : Numeric
         Source X coordinate

    sy : Numeric
         Source y coordinate

    dx : Numeric
         The desintation x coordinate

    dy : Numeric
         The destination y coordinate

    s_img : GeoDataset
            The source image GeoDataset

    d_img : GeoDataset
            The destination image GeoDataset

    image_size : tuple
                 (xsize, ysize) of the image that is searched within (this should be larger
                 than the template size)

    template_size : tuple
                    (xsize, ysize) of the template to iterate over the image in order
                    to identify the area(s) of highest correlation.

    func : callable
           The function used to pattern match

    verbose : bool
              If true, generate plots of the matches

    Returns
    -------
    x_shift : float
               Shift in the x-dimension

    y_shift : float
               Shift in the y-dimension

    strength : float
               Strength of the correspondence in the range [-1, 1]

    corrmap : ndarray
              An n,m array of correlation coefficients

    See Also
    --------
    autocnet.matcher.naive_template.pattern_match : for the kwargs that can be passed to the matcher
    autocnet.matcher.naive_template.pattern_match_autoreg : for the jwargs that can be passed to the autoreg style matcher
    """
    image_size = check_image_size(image_size)
    template_size = check_image_size(template_size)

    template_size_x = template_size[0]
    template_size_y = template_size[1]

    s_roi = roi.Roi(s_img, sx, sy, size_x=image_size[0], size_y=image_size[1])
    d_roi = roi.Roi(d_img, dx, dy, size_x=template_size_x, size_y=template_size_y)

    if d_roi.variance == 0:
        return [None] * 4

    if not s_roi.is_valid or not d_roi.is_valid:
        return [None] * 4

    try:
        s_image_dtype = isis.isis2np_types[pvl.load(s_img.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    except:
        s_image_dtype = None

    try:
        d_template_dtype = isis.isis2np_types[pvl.load(d_img.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    except:
        d_template_dtype = None

    s_image = s_roi.array
    d_template = d_roi.array

    if (s_image is None) or (d_template is None):
        return None, None, None, None

    # Apply the matcher function
    shift_x, shift_y, metrics, corrmap = func(d_template, s_image, **kwargs)

    if verbose:
        fig, axs = plt.subplots(1, 3, figsize=(20,10))
        axs[0].imshow(s_image, cmap='Greys')
        axs[1].imshow(d_template, cmap='Greys')
        axs[3].imshow(corrmap)
        plt.show()

    # Hard check here to see if we are on the absolute edge of the template
    max_coord = np.unravel_index(corrmap.argmax(), corrmap.shape)[::-1]
    if 0 in max_coord or corrmap.shape[0]-1 == max_coord[0] or corrmap.shape[1]-1 == max_coord[1]:
        warnings.warn('Maximum correlation is at the edge of the template. Results are ambiguous.', UserWarning)
        return [None] * 4

    # Apply the shift to the center of the ROI object
    dx = d_roi.x - shift_x
    dy = d_roi.y - shift_y

    return dx, dy, metrics, corrmap

def subpixel_ciratefi(sx, sy, dx, dy, s_img, d_img, search_size=251, template_size=51, **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.

    Parameters
    ----------
    sx : numeric
         The x position of the center of the template to be matched to
    sy : numeric
         The y position of the center of the template to be matched to
    dx : numeric
         The x position of the center of the search to be matched from
    dy : numeric
         The y position of the center of the search to be matched to
    s_img : object
            A plio geodata object from which the template is extracted
    d_img : object
            A plio geodata object from which the search is extracted
    search_size : int
                  An odd integer for the size of the search image
    template_size : int
                    A odd integer for the size of the template that is iterated
                    over the search images

    Returns
    -------
    x_offset : float
               Shift in the x-dimension

    y_offset : float
               Shift in the y-dimension

    strength : float
               Strength of the correspondence in the range [-1, 1]
    """
    t_roi = roi.Roi(d_img, dx, dy,
                              size_x=template_size, size_y=template_size)
    s_roi = roi.Roi(s_img, sx, sy,
                                size_x=search_size, size_y=search_size)
    template = t_roi.array
    search = s_roi.array

    if template is None or search is None:
        return None, None, None

    x_offset, y_offset, strength = ciratefi.ciratefi(template, search, **kwargs)
    dx += (x_offset + t_roi.axr)
    dy += (y_offset + t_roi.ayr)
    return dx, dy, strength

def iterative_phase(sx, sy, dx, dy, s_img, d_img, size=(51, 51), reduction=11, convergence_threshold=1.0, max_dist=50, **kwargs):
    """
    Iteratively apply a subpixel phase matcher to source (s_img) and destination (d_img)
    images. The size parameter is used to set the initial search space. The algorithm
    is recursively applied to reduce the total search space by reduction until the convergence criteria
    are met. Convergence is defined as the point at which the computed shifts (x_shift,y_shift) are
    less than the convergence_threshold. In instances where the size is reducted to 1 pixel the
    algorithm terminates and returns None.

    Parameters
    ----------
    sx : numeric
         The x position of the center of the template to be matched to
    sy : numeric
         The y position of the center of the template to be matched to
    dx : numeric
         The x position of the center of the search to be matched from
    dy : numeric
         The y position of the center of the search to be matched to
    s_img : object
            A plio geodata object from which the template is extracted
    d_img : object
            A plio geodata object from which the search is extracted
    size : tuple
           Size of the template in the form (x,y)
    reduction : int
                With each recursive call to this func, the size is reduced by this amount
    convergence_threshold : float
                            The value under which the result can shift in the x and y directions to force a break

    Returns
    -------
    dx : float
         The new x value for the match in the destination (d) image
    dy : float
         The new y value for the match in the destination (d) image
    metrics : tuple
              A tuple of metrics. In the case of the phase matcher this are difference
              and RMSE in the phase dimension.

    See Also
    --------
    subpixel_phase : the function that applies a single iteration of the phase matcher
    """

    # get initial destination location
    dsample = dx
    dline = dy

    while True:
        shifted_dx, shifted_dy, metrics, _ = subpixel_phase(sx, sy, dx, dy, s_img, d_img, image_size=size, **kwargs)

        # Compute the amount of move the matcher introduced
        delta_dx = abs(shifted_dx - dx)
        delta_dy = abs(shifted_dy - dy)
        dx = shifted_dx
        dy = shifted_dy

        # Break if the solution has converged
        size = (size[0] - reduction, size[1] - reduction)
        dist = np.linalg.norm([dsample-dx, dline-dy])

        if min(size) < 1:
            return None, None, (None, None)
        if delta_dx <= convergence_threshold and\
           delta_dy<= convergence_threshold and\
           abs(dist) <= max_dist:
           break

    return dx, dy, metrics

'''def estimate_affine_transformation(destination_coordinates, source_coordinates):
    """
    Given a set of destination control points compute the affine transformation
    required to project the source control points into the destination.

    Parameters
    ----------
    destination_coordinates : array-like
                              An n,2 data structure containing the destination control points

    source_coordinates : array-like
                         An n,2 data structure containing the source control points

    Returns
    -------
     : object
       An skimage affine transform object
    """
    destination_coordinates = np.asarray(destination_coordinates)
    source_coordinates = np.asarray(source_coordinates)

    return tf.estimate_transform('affine', destination_coordinates, source_coordinates)
'''
def geom_match_simple(reference_image,
                       moving_image,
                       bcenter_x,
                       bcenter_y,
                       size_x=60,
                       size_y=60,
                       match_func="classic",
                       match_kwargs={"image_size":(101,101), "template_size":(31,31)},
                       preprocess=None,
                       verbose=False):
    """
    Propagates a source measure into destination images and then perfroms subpixel registration.
    Measure creation is done by projecting the (lon, lat) associated with the source measure into the
    destination image. The created measure is then matched to the source measure using a quick projection
    of the destination image into source image space (using an affine transformation) and a naive
    template match with optional phase template match.

    This version projects the entirity of the input cube onto the base

    Parameters
    ----------
    base_cube:  plio.io.io_gdal.GeoDataset
                source image
    input_cube: plio.io.io_gdal.GeoDataset
                destination image; gets matched to the source image
    bcenter_x:  int
                sample location of source measure in base_cube
    bcenter_y:  int
                line location of source measure in base_cube
    size_x:     int
                half-height of the subimage used in the affine transformation
    size_y:     int
                half-width of the subimage used in affine transformation
    template_kwargs: dict
                     contains keywords necessary for autocnet.matcher.subpixel.subpixel_template
    phase_kwargs:    dict
                     contains kwargs for autocnet.matcher.subpixel.subpixel_phase
    verbose:    boolean
                indicates level of print out desired. If True, two subplots are output; the first subplot contains
                the source subimage and projected destination subimage, the second subplot contains the registered
                measure's location in the base subimage and the unprojected destination subimage with the corresponding
                template metric correlation map.
    Returns
    -------
    sample: int
            sample of new measure in destination image space
    line:   int
            line of new measures in destination image space
    dist:   np.float or tuple of np.float
            distance matching algorithm moved measure
            if template matcher only (default): returns dist_template
            if template and phase matcher:      returns (dist_template, dist_phase)
    metric: np.float or tuple of np.float
            matching metric output by the matcher
            if template matcher only (default): returns maxcorr
            if template and phase matcher:      returns (maxcorr, perror, pdiff)
    temp_corrmap: np.ndarray
            correlation map of the naive template matcher
    See Also
    --------
    autocnet.matcher.subpixel.subpixel_template: for list of kwargs that can be passed to the matcher
    autocnet.matcher.subpixel.subpixel_phase: for list of kwargs that can be passed to the matcher
    """
    t1 = time.time()
    if not isinstance(reference_image, GeoDataset):
        raise TypeError("reference_image must be a GeoDataset obj")
    if not isinstance(moving_image, GeoDataset):
        raise TypeError("moving_image must be a GeoDataset obj")

    # Parse the match_func into a function if it comes in as a string
    if not callable(match_func):
        match_func = check_match_func(match_func)

    # Estimate the transformation between the two images
    affine = estimate_affine_transformation(reference_image, moving_image, bcenter_x, bcenter_y)
    t2 = time.time()
    print(f'Estimation of the transformation took {t2-t1} seconds.')



    # Read the arrays with the correct dtype - needs to be in ROI object for dtype checking.
    base_type = isis.isis2np_types[pvl.load(base_cube.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    base_arr = base_cube.read_array(dtype=base_type)

    dst_type = isis.isis2np_types[pvl.load(input_cube.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    dst_arr = input_cube.read_array(dtype=dst_type)

    box = (0, 0, max(dst_arr.shape[1], base_arr.shape[1]), max(dst_arr.shape[0], base_arr.shape[0]))
    dst_arr = np.array(Image.fromarray(dst_arr).crop(box))

    dst_arr = tf.warp(dst_arr, affine, order=3)
    t3 = time.time()
    print(f'Affine warp took {t3-t2} seconds.')
    if verbose:
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title("Base")
        axs[0].imshow(roi.Roi(bytescale(base_arr, cmin=0), bcenter_x, bcenter_y, 25, 25).array, cmap="Greys_r")
        axs[1].set_title("Projected Image")
        axs[1].imshow(roi.Roi(bytescale(dst_arr, cmin=0), bcenter_x, bcenter_y, 25, 25).array, cmap="Greys_r")
        plt.show()
    # Run through one step of template matching then one step of phase matching
    # These parameters seem to work best, should pass as kwargs later
    #print('dst_arr mean: ', dst_arr.mean())
    #print(f'dst_arr mm: {dst_arr.min()} {dst_arr.max()}')
    #print(f'special pixels: ', isis.get_isis_special_pixels(dst_arr))

    if preprocess:
        base_arr, dst_arr = preprocess(base_arr, dst_arr)

    restemplate = match_func(bcenter_x, bcenter_y, bcenter_x, bcenter_y, base_arr, dst_arr, **match_kwargs)
    t4 = time.time()
    print(f'Matching took {t4-t3} seconds')

    try:
        x,y,maxcorr,temp_corrmap = restemplate
    except:
        # did not return a corrmap
        x,y,maxcorr = restemplate
        temp_corrmap = np.empty((size_x, size_y))
        temp_corrmap[:] = np.nan

    if x is None or y is None:
        return None, None, None, None, None
    metric = maxcorr
    sample, line = affine([x, y])[0]
    dist = np.linalg.norm([bcenter_x-x, bcenter_y-y])

    if verbose:
        fig, axs = plt.subplots(2, 3)
        fig.set_size_inches((30,30))

        oarr = roi.Roi(input_cube.read_array(), sample, line, 150, 150).array
        axs[0][2].imshow(bytescale(oarr, cmin=0), cmap="Greys_r")
        axs[0][2].axhline(y=oarr.shape[1]/2, color="red", linestyle="-", alpha=1)
        axs[0][2].axvline(x=oarr.shape[1]/2, color="red", linestyle="-", alpha=1)
        axs[0][2].set_title("Original Registered Image")

        barr = roi.Roi(base_arr, bcenter_x, bcenter_y, size_x, size_y).array
        axs[0][0].imshow(bytescale(barr, cmin=0), cmap="Greys_r")
        axs[0][0].axhline(y=barr.shape[1]/2, color="red", linestyle="-", alpha=1)
        axs[0][0].axvline(x=barr.shape[1]/2, color="red", linestyle="-", alpha=1)
        axs[0][0].set_title("Base")

        darr = roi.Roi(dst_arr, x, y, size_x, size_y).array
        axs[0][1].imshow(bytescale(darr, cmin=0), cmap="Greys_r")
        axs[0][1].axhline(y=darr.shape[1]/2, color="red", linestyle="-", alpha=1)
        axs[0][1].axvline(x=darr.shape[1]/2, color="red", linestyle="-", alpha=1)
        axs[0][1].set_title("Projected Registered Image")

        #axs[1][0].imshow(bytescale(darr.astype("f")/barr.astype("f")), cmap="Greys_r", alpha=.6)
        axs[1][0].imshow(bytescale(barr.astype("f")), cmap="Blues", alpha=1.0)
        axs[1][0].imshow(bytescale(darr.astype("f")), cmap="Pastel2", alpha=0.5)
        axs[1][0].axhline(y=barr.shape[1]/2, color="red", linestyle="-", alpha=.5)
        axs[1][0].axvline(x=barr.shape[1]/2, color="red", linestyle="-", alpha=.5)
        axs[1][0].set_title("overlap")

        pcm = axs[1][1].imshow(temp_corrmap**2, interpolation=None, cmap="coolwarm")
        plt.show()

    return sample, line, dist, metric, temp_corrmap

def geom_match_classic(base_cube,
                       input_cube,
                       bcenter_x,
                       bcenter_y,
                       size_x=60,
                       size_y=60,
                       template_kwargs={"image_size":(59,59), "template_size":(31,31)},
                       phase_kwargs=None,
                       verbose=True):
    """
    Propagates a source measure into destination images and then perfroms subpixel registration.
    Measure creation is done by projecting the (lon, lat) associated with the source measure into the
    destination image. The created measure is then matched to the source measure using a quick projection
    of the destination image into source image space (using an affine transformation) and a naive
    template match with optional phase template match.
    Parameters
    ----------
    base_cube:  plio.io.io_gdal.GeoDataset
                source image
    input_cube: plio.io.io_gdal.GeoDataset
                destination image; gets matched to the source image
    bcenter_x:  int
                sample location of source measure in base_cube
    bcenter_y:  int
                line location of source measure in base_cube
    size_x:     int
                half-height of the subimage used in the affine transformation
    size_y:     int
                half-width of the subimage used in affine transformation
    template_kwargs: dict
                     contains keywords necessary for autocnet.matcher.subpixel.subpixel_template
    phase_kwargs:    dict
                     contains kwargs for autocnet.matcher.subpixel.subpixel_phase
    verbose:    boolean
                indicates level of print out desired. If True, two subplots are output; the first subplot contains
                the source subimage and projected destination subimage, the second subplot contains the registered
                measure's location in the base subimage and the unprojected destination subimage with the corresponding
                template metric correlation map.
    Returns
    -------
    sample: int
            sample of new measure in destination image space
    line:   int
            line of new measures in destination image space
    dist:   np.float or tuple of np.float
            distance matching algorithm moved measure
            if template matcher only (default): returns dist_template
            if template and phase matcher:      returns (dist_template, dist_phase)
    metric: np.float or tuple of np.float
            matching metric output by the matcher
            if template matcher only (default): returns maxcorr
            if template and phase matcher:      returns (maxcorr, perror, pdiff)
    temp_corrmap: np.ndarray
            correlation map of the naive template matcher
    See Also
    --------
    autocnet.matcher.subpixel.subpixel_template: for list of kwargs that can be passed to the matcher
    autocnet.matcher.subpixel.subpixel_phase: for list of kwargs that can be passed to the matcher
    """

    if not isinstance(input_cube, GeoDataset):
        raise Exception("input cube must be a geodataset obj")
    if not isinstance(base_cube, GeoDataset):
        raise Exception("match cube must be a geodataset obj")

    base_startx = int(bcenter_x - size_x)
    base_starty = int(bcenter_y - size_y)
    base_stopx = int(bcenter_x + size_x)
    base_stopy = int(bcenter_y + size_y)

    image_size = input_cube.raster_size
    match_size = base_cube.raster_size

    # for now, require the entire window resides inside both cubes.
    if base_stopx > match_size[0]:
        raise Exception(f"Window: {base_stopx} > {match_size[0]}, center: {bcenter_x},{bcenter_y}")
    if base_startx < 0:
        raise Exception(f"Window: {base_startx} < 0, center: {bcenter_x},{bcenter_y}")
    if base_stopy > match_size[1]:
        raise Exception(f"Window: {base_stopy} > {match_size[1]}, center: {bcenter_x},{bcenter_y} ")
    if base_starty < 0:
        raise Exception(f"Window: {base_starty} < 0, center: {bcenter_x},{bcenter_y}")

    # specifically not putting this in a try/except, this should never fail
    mlon, mlat = spatial.isis.image_to_ground(base_cube.file_name, bcenter_x, bcenter_y)
    center_x, center_y = spatial.isis.ground_to_image(input_cube.file_name, mlon, mlat)

    base_corners = [(base_startx,base_starty),
                    (base_startx,base_stopy),
                    (base_stopx,base_stopy),
                    (base_stopx,base_starty)]

    dst_corners = []
    for x,y in base_corners:
        try:
            lon, lat = spatial.isis.image_to_ground(base_cube.file_name, x, y)
            dst_corners.append(
                spatial.isis.ground_to_image(input_cube.file_name, lon, lat)
            )
        except CalledProcessError as e:
            if 'Requested position does not project in camera model' in e.stderr:
                print(f'Skip geom_match; Region of interest corner located at ({lon}, {lat}) does not project to image {input_cube.base_name}')
                return None, None, None, None, None

    base_gcps = np.array([*base_corners])
    base_gcps[:,0] -= base_startx
    base_gcps[:,1] -= base_starty

    dst_gcps = np.array([*dst_corners])
    start_x = dst_gcps[:,0].min()
    start_y = dst_gcps[:,1].min()
    stop_x = dst_gcps[:,0].max()
    stop_y = dst_gcps[:,1].max()
    dst_gcps[:,0] -= start_x
    dst_gcps[:,1] -= start_y

    affine = tf.estimate_transform('affine', np.array([*base_gcps]), np.array([*dst_gcps]))

    base_pixels = list(map(int, [base_corners[0][0], base_corners[0][1], size_x*2, size_y*2]))
    base_type = isis.isis2np_types[pvl.load(base_cube.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    base_arr = base_cube.read_array(pixels=base_pixels, dtype=base_type)

    dst_pixels = list(map(int, [start_x, start_y, stop_x-start_x, stop_y-start_y]))
    dst_type = isis.isis2np_types[pvl.load(input_cube.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    dst_arr = input_cube.read_array(pixels=dst_pixels, dtype=dst_type)

    dst_arr = tf.warp(dst_arr, affine)
    dst_arr = dst_arr[:size_y*2, :size_x*2]

    if verbose:
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title("Base")
        axs[0].imshow(bytescale(base_arr), cmap="Greys_r")
        axs[1].set_title("Projected Image")
        axs[1].imshow(bytescale(dst_arr), cmap="Greys_r")
        plt.show()

    # Run through one step of template matching then one step of phase matching
    # These parameters seem to work best, should pass as kwargs later
    restemplate = subpixel_template_classic(size_x, size_y, size_x, size_y, bytescale(base_arr), bytescale(dst_arr), **template_kwargs)

    x,y,maxcorr,temp_corrmap = restemplate
    if x is None or y is None:
        return None, None, None, None, None
    metric = maxcorr
    sample, line = affine([x, y])[0]
    sample += start_x
    line += start_y
    dist = np.linalg.norm([center_x-sample, center_y-line])

    if verbose:
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches((30,30))
        darr = roi.Roi(input_cube.read_array(dtype=dst_type), sample, line, 100, 100).array
        axs[1].imshow(darr, cmap="Greys_r")
        axs[1].scatter(x=[darr.shape[1]/2], y=[darr.shape[0]/2], s=10, c="red")
        axs[1].set_title("Original Registered Image")

        axs[0].imshow(base_arr, cmap="Greys_r")
        axs[0].scatter(x=[base_arr.shape[1]/2], y=[base_arr.shape[0]/2], s=10, c="red")
        axs[0].set_title("Base")

        pcm = axs[2].imshow(temp_corrmap**2, interpolation=None, cmap="coolwarm")
        plt.show()

    return sample, line, dist, metric, temp_corrmap

def geom_match(destination_cube,
               source_cube,
               bcenter_x,
               bcenter_y,
               template_kwargs={"image_size":(59,59), "template_size":(31,31)},
               verbose=True):
    """
    Propagates a source measure into destination images and then perfroms subpixel registration.
    Measure creation is done by projecting the (lon, lat) associated with the source measure into the
    destination image. The created measure is then matched to the source measure using a quick projection
    of the destination image into source image space (using an affine transformation) and a naive
    template match with optional phase template match.

    Parameters
    ----------
    destination_cube:  plio.io.io_gdal.GeoDataset
                       The image to be matched to

    source_cube: plio.io.io_gdal.GeoDataset
                 The image that is transformed and matched into the destination_cube

    bcenter_x:  int
                sample location of source measure in base_cube

    bcenter_y:  int
                line location of source measure in base_cube

    size_x:     int
                half-height of the subimage used in the affine transformation

    size_y:     int
                half-width of the subimage used in affine transformation

    template_kwargs: dict
                     contains keywords necessary for autocnet.matcher.subpixel.subpixel_template

    verbose:    boolean
                indicates level of print out desired. If True, two subplots are output; the first subplot contains
                the source subimage and projected destination subimage, the second subplot contains the registered
                measure's location in the base subimage and the unprojected destination subimage with the corresponding
                template metric correlation map.

    Returns
    -------
    sample: int
            sample of new measure in destination image space

    line:   int
            line of new measures in destination image space

    dist:   np.float or tuple of np.float
            distance matching algorithm moved measure
            if template matcher only (default): returns dist_template
            if template and phase matcher:      returns (dist_template, dist_phase)

    metric: np.float or tuple of np.float
            matching metric output by the matcher
            if template matcher only (default): returns maxcorr
            if template and phase matcher:      returns (maxcorr, perror, pdiff)

    temp_corrmap: np.ndarray
                  correlation map of the naive template matcher

    See Also
    --------
    autocnet.matcher.subpixel.subpixel_template: for list of kwargs that can be passed to the matcher
    autocnet.matcher.subpixel.subpixel_phase: for list of kwargs that can be passed to the matcher

    """

    if not isinstance(source_cube, GeoDataset):
        raise Exception("source cube must be a geodataset obj")

    if not isinstance(destination_cube, GeoDataset):
        raise Exception("destination cube must be a geodataset obj")

    destination_size_x = template_kwargs['image_size'][0]
    destination_size_y = template_kwargs['image_size'][1]

    destination_startx = int(bcenter_x - destination_size_x)
    destination_starty = int(bcenter_y - destination_size_y)
    destination_stopx = int(bcenter_x + destination_size_x)
    destination_stopy = int(bcenter_y + destination_size_y)

    image_size = source_cube.raster_size
    match_size = destination_cube.raster_size

    # for now, require the entire window resides inside both cubes.
    if destination_stopx > match_size[0]:
        raise Exception(f"Window: {destination_stopx} > {match_size[0]}, center: {bcenter_x},{bcenter_y}")
    if destination_startx < 0:
        raise Exception(f"Window: {destination_startx} < 0, center: {bcenter_x},{bcenter_y}")
    if destination_stopy > match_size[1]:
        raise Exception(f"Window: {destination_stopy} > {match_size[1]}, center: {bcenter_x},{bcenter_y} ")
    if destination_starty < 0:
        raise Exception(f"Window: {destination_starty} < 0, center: {bcenter_x},{bcenter_y}")

    destination_corners = [(destination_startx,destination_starty),
                    (destination_startx,destination_stopy),
                    (destination_stopx,destination_stopy),
                    (destination_stopx,destination_starty)]

    # specifically not putting this in a try/except, this should never fail
    # 07/28 - putting it in a try/except because of how we ground points
    # Transform from the destination center to the source_cube center
    try:
        mlon, mlat = spatial.isis.image_to_ground(destination_cube.file_name, bcenter_x, bcenter_y)
        center_x, center_y = spatial.isis.ground_to_image(source_cube.file_name, mlon, mlat)
    except CalledProcessError as e:
            if 'Requested position does not project in camera model' in e.stderr:
                print(f'Skip geom_match; Region of interest center located at ({mlon}, {mlat}) does not project to image {source_cube.base_name}')
                print('This should only appear when propagating ground points')
                return None, None, None, None, None

    # Compute the mapping between the destination corners and the source_cube corners in
    # order to estimate an affine transformation
    source_corners = []
    for x,y in destination_corners:
        try:
            lon, lat = spatial.isis.image_to_ground(destination_cube.file_name, x, y)
            source_corners.append(
                spatial.isis.ground_to_image(source_cube.file_name, lon, lat)
            )
        except CalledProcessError as e:
            if 'Requested position does not project in camera model' in e.stderr:
                print(f'Skip geom_match; Region of interest corner located at ({lon}, {lat}) does not project to image {source_cube.base_name}')
                return None, None, None, None, None


    # Estimate the transformation
    affine = estimate_affine_transformation(destination_corners, source_corners)

    # Apply the subpixel matcher with an affine transformation
    restemplate = subpixel_transformed_template(bcenter_x, bcenter_y,
                                                center_x, center_y,
                                                destination_cube, source_cube,
                                                affine,
                                                verbose=verbose,
                                                **template_kwargs)

    x, y, metric, corrmap = restemplate

    if x is None or y is None:
        return None, None, None, None, None

    dist = np.linalg.norm([center_x-x, center_y-y])
    return x, y, dist, metric, corrmap

def subpixel_register_measure(measureid,
                              subpixel_template_kwargs={},
                              size_x=100, size_y=100,
                              cost_func=lambda x,y: 1/x**2 * y,
                              threshold=0.005,
                              ncg=None,
                              **kwargs):
    """
    Given a measure, subpixel register to the reference measure of its associated point.

    Parameters
    ----------
    ncg : obj
          the network candidate graph that the point is associated with; used for
          the DB session that is able to access the point.

    measureid : int or obj
              The identifier of the measure in the DB or a Measures object

    subpixel_template_kwargs : dict
                               Any keyword arguments passed to the template matcher

    cost : func
           A generic cost function accepting two arguments (x,y), where x is the
           distance that a point has shifted from the original, sensor identified
           intersection, and y is the correlation coefficient coming out of the
           template matcher.

    threshold : numeric
                measures with a cost <= the threshold are marked as ignore=True in
                the database.
    """

    if isinstance(measureid, Measures):
        measureid = measureid.id

    result = {'measureid':measureid,
              'status':''}

    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    with ncg.session_scope() as session:
        # Setup the measure that is going to be matched
        destination = session.query(Measures).filter(Measures.id == measureid).one()
        destinationimageid = destination.imageid
        destinationimage = session.query(Images).filter(Images.id == destinationimageid).one()
        destination_node = NetworkNode(node_id=destinationimageid, image_path=destinationimage.path)

        # Get the point id and set up the reference measure
        pointid = destination.pointid
        measures = session.query(Measures).filter(Measures.pointid==pointid).order_by(Measures.id).all()

        reference_index = measures[0].reference_index
        source = measures[reference_index]

        source.template_metric = 1
        source.template_shift = 0
        source.phase_error = 0
        source.phase_diff = 0
        source.phase_shift = 0
        source.weight = 1

        sourceid = source.imageid
        sourceimage = session.query(Images).filter(Images.id == sourceid).one()
        source_node = NetworkNode(node_id=sourceid, image_path=sourceimage.path)

        resultlog = []
        print(f'Attempting to subpixel register measure {measureid}: ({pointid}, {destinationimage.name})')
        currentlog = {'measureid': measureid,
                      'status': ''}

        if source.measureid == measureid:
            currentlog['status'] = f'Unable to register this measure. Measure {measureid} is the reference measure.'
            return resultlog

        try:
            new_x, new_y, dist, metric = geom_match_simple(source_node.geodata, destination_node.geodata,
                                                            source.sample, source.line,
                                                            match_func=match_func,
                                                            template_kwargs=subpixel_template_kwargs)
        except Exception as e:
            print(f'geom_match failed on measure {measureid} with exception -> {e}')
            destination.ignore = True # geom_match failed
            currentlog['status'] = f"Failed to register measure {measureid}"
            resultlog.append(currentlog)
            return resultlog

        if new_x == None or new_y == None:
            destination.ignore = True # Unable to geom match
            currentlog['status'] = 'Failed to geom match.'
            resultlog.append(currentlog)
            return resultlog

        destination.template_metric = metric
        destination.template_shift = dist

        cost = cost_func(destination.template_shift, destination.template_metric)

        if cost <= threshold:
            destination.ignore = True # Threshold criteria not met
            currentlog['status'] = f'Cost failed. Distance shifted: {destination.template_shift}. Metric: {destination.template_metric}.'
            resultlog.append(currentlog)
            return resultlog

        # Update the measure
        destination.sample = new_x
        destination.line = new_y
        destination.weight = cost
        destination.choosername = 'subpixel_register_measure'

        # In case this is a second run, set the ignore to False if this
        # measures passed. Also, set the source measure back to ignore=False
        destination.ignore = False
        source.ignore = False
        currentlog['status'] = f'Success.'
        resultlog.append(currentlog)


    return resultlog

def subpixel_register_point(pointid,
                            cost_func=lambda x,y: 1/x**2 * y,
                            threshold=0.005,
                            ncg=None,
                            geom_func='simple',
                            match_func='classic',
                            match_kwargs={},
                            use_cache=False,
                            verbose=False,
                            chooser='subpixel_register_point',
                            **kwargs):

    """
    Given some point, subpixel register all of the measures in the point to the
    first measure.

    Parameters
    ----------
    pointid : int or obj
              The identifier of the point in the DB or a Points object

    cost_func : func
                A generic cost function accepting two arguments (x,y), where x is the
                distance that a point has shifted from the original, sensor identified
                intersection, and y is the correlation coefficient coming out of the
                template matcher.

    threshold : numeric
                measures with a cost <= the threshold are marked as ignore=True in
                the database.
    ncg : obj
          the network candidate graph that the point is associated with; used for
          the DB session that is able to access the point.

    geom_func : callable
                function used to tranform the source and/or destination image before
                running a matcher.

    match_func : callable
                 subpixel matching function to use registering measures

    use_cache : bool
                If False (default) this func opens a database session and writes points
                and measures directly to the respective tables. If True, this method writes
                messages to the point_insert (defined in ncg.config) redis queue for
                asynchronous (higher performance) inserts.
    """

    geom_func=geom_func.lower()
    match_func=match_func.lower()

    print(f"Using {geom_func} with the {match_func} matcher.")

    match_func = check_match_func(match_func)
    geom_func = check_geom_func(geom_func)

    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    if isinstance(pointid, Points):
        pointid = pointid.id

    t1 = time.time()
    with ncg.session_scope() as session:
        measures = session.query(Measures).filter(Measures.pointid == pointid).order_by(Measures.id).all()
        point = session.query(Points).filter(Points.id == pointid).one()
        reference_index = point.reference_index
        t2 = time.time()
        print(f'Query took {t2-t1} seconds to find the measures and reference measure.')
        # Get the reference measure. Previously this was index 0, but now it is a database tracked attribute
        source = measures[reference_index]

        print(f'Using measure {source.id} on image {source.imageid}/{source.serial} as the reference.')
        print(f'Measure reference index is: {reference_index}')
        source.template_metric = 1
        source.template_shift = 0
        source.phase_error = 0
        source.phase_diff = 0
        source.phase_shift = 0

        sourceid = source.imageid
        sourceres = session.query(Images).filter(Images.id == sourceid).one()
        source_node = NetworkNode(node_id=sourceid, image_path=sourceres.path)
        source_node.parent = ncg
        t3 = time.time()
        print(f'Query for the image to use as source took {t3-t2} seconds.')
        print(f'Attempting to subpixel register {len(measures)-1} measures for point {pointid}')
        nodes = {}
        for measure in measures:
            res = session.query(Images).filter(Images.id == measure.imageid).one()
            nodes[measure.imageid] = NetworkNode(node_id=measure.imageid, image_path=res.path)
        session.expunge_all()

    resultlog = []
    updated_measures = []
    for i, measure in enumerate(measures):
        if i == reference_index:
            continue

        currentlog = {'measureid':measure.id,
                    'status':''}
        cost = None
        destinationid = measure.imageid

        destination_node = nodes[measure.imageid]

        print('geom_match image:', destination_node['image_path'])
        print('geom_func', geom_func)
        try:
            # new geom_match has a incompatible API, until we decide on one, put in if.
            if (geom_func == geom_match):
               new_x, new_y, dist, metric,  _ = geom_func(source_node.geodata, destination_node.geodata,
                                                    source.apriorisample, source.aprioriline,
                                                    template_kwargs=match_kwargs,
                                                    verbose=verbose)
            else:
                new_x, new_y, dist, metric,  _ = geom_func(source_node.geodata, destination_node.geodata,
                                                    source.apriorisample, source.aprioriline,
                                                    match_func=match_func,
                                                    match_kwargs=match_kwargs,
                                                    verbose=verbose)
        except Exception as e:
            print(f'geom_match failed on measure {measure.id} with exception -> {e}')
            currentlog['status'] = f"geom_match failed on measure {measure.id}"
            resultlog.append(currentlog)
            if measure.weight is None:
                measure.ignore = True # Geom match failed and no previous sucesses
            updated_measures.append(measure)
            continue

        if new_x == None or new_y == None:
            currentlog['status'] = f'Failed to register measure {measure.id}.'
            resultlog.append(currentlog)
            if measure.weight is None:
                measure.ignore = True # Unable to geom match and no previous sucesses
            updated_measures.append(measure)
            continue

        measure.template_metric = metric
        measure.template_shift = dist

        cost = cost_func(measure.template_shift, measure.template_metric)

        print(f'Current Cost: {cost},  Current Weight: {measure.weight}')

        # Check to see if the cost function requirement has been met
        if measure.weight and cost <= measure.weight:
            currentlog['status'] = f'Previous match provided better correlation. {measure.weight} > {cost}.'
            resultlog.append(currentlog)
            updated_measures.append(measure)
            continue

        if cost <= threshold:
            currentlog['status'] = f'Cost failed. Distance calculated: {measure.template_shift}. Metric calculated: {measure.template_metric}.'
            resultlog.append(currentlog)
            updated_measures.append(measure)
            if measure.weight is None:
                measure.ignore = True # Threshold criteria not met and no previous sucesses
            continue

        # Update the measure
        measure.sample = new_x
        measure.line = new_y
        measure.weight = cost
        measure.choosername = chooser

        # In case this is a second run, set the ignore to False if this
        # measures passed. Also, set the source measure back to ignore=False
        measure.ignore = False
        # Maybe source?
        source.ignore = False
        updated_measures.append(measure)
        currentlog['status'] = f'Success. Distance shifted: {measure.template_shift}. Metric: {measure.template_metric}.'
        resultlog.append(currentlog)

    # Once here, update the source measure (possibly back to ignore=False)
    updated_measures.append(source)

    if use_cache:
        t4 = time.time()
        ncg.redis_queue.rpush(ncg.measure_update_queue,
                              *[json.dumps(measure.to_dict(_hide=[]), cls=JsonEncoder) for measure in updated_measures])
        ncg.redis_queue.incr(ncg.measure_update_counter, amount=len(updated_measures))
        t5 = time.time()
        print(f'Cache load took {t5-t4} seconds')
    else:
        t4 = time.time()
        # Commit the updates back into the DB
        with ncg.session_scope() as session:
            for m in updated_measures:
                ins = inspect(m)
                session.add(m)
        t5 = time.time()
        print(f'Database update took {t5-t4} seconds.')
    return resultlog

def subpixel_register_points(subpixel_template_kwargs={'image_size':(251,251)},
                             cost_kwargs={},
                             threshold=0.005,
                             Session=None):
    """
    Serial subpixel registration of all of the points in a given DB table.

    Parameters
    ----------
    Session : obj
              A SQLAlchemy Session factory.

    pointid : int
              The identifier of the point in the DB

    subpixel_template_kwargs : dict
                               Ay keyword arguments passed to the template matcher

    cost : func
           A generic cost function accepting two arguments (x,y), where x is the
           distance that a point has shifted from the original, sensor identified
           intersection, and y is the correlation coefficient coming out of the
           template matcher.

    threshold : numeric
                measures with a cost <= the threshold are marked as ignore=True in
                the database.
    """
    if not Session:
        raise BrokenPipeError('This func requires a database session.')
    session = Session()
    pointids = [point.id for point in session.query(Points)]
    session.close()
    for pointid in pointids:
        subpixel_register_point(pointid,
                                subpixel_template_kwargs=subpixel_template_kwargs,
                                **cost_kwargs)

def register_to_base(pointid,
                     base_image,
                     cost_func=lambda x, y: y == np.max(x),
                     ncg=None,
                     geom_func='simple',
                     geom_kwargs={"size_x": 16, "size_y": 16},
                     match_func='classic',
                     match_kwargs={},
                     verbose=False,
                     **kwargs):
    """
    """

    if not ncg.Session:
     raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    geom_func = check_geom_func(geom_func)
    match_func = check_match_func(match_func)
    session = ncg.Session()

    if isinstance(base_image, str):
        base_image = GeoDataset(base_image)

    if isinstance(pointid, Points):
        point = pointid
        pointid = pointid.id

    with ncg.session_scope() as session:
        if isinstance(pointid, Points):
            point = pointid
            pointid = point.id
        else:
            point = session.query(Points).filter(Points.id == pointid).one()

        # Get all of the measures associated with the given point
        measures = point.measures

        # Attempt to project the point into the base image
        bpoint = spatial.isis.point_info(base_image.file_name, point.geom.x, point.geom.y, 'ground')
        if bpoint is None:
            print('unable to find point in ground image')
            # Need to set the point to False
            return
        bline = bpoint.get('Line')
        bsample = bpoint.get('Sample')

        # Setup a cache so that we can get the file handles one time instead of
        # once per measure in the measures list.
        image_cache = {}

        # list of matching results in the format:
        # [measure_id, measure_index, x_offset, y_offset, offset_magnitude]
        match_results = []

        # Step over all of the measures (images) that are not the base image
        for measure_index, measure in enumerate(measures):
            res = session.query(Images).filter(Images.id == measure.imageid).one()
            try:
                measure_image = image_cache[res.id]
            except:
                measure_image = GeoDataset(measure_image.path)
                image_cache[res.id] = measure_image

            # Attempt to match the base
            try:
                print(f'prop point: base_image: {base_image}')
                print(f'prop point: dest_image: {measure_image}')
                print(f'prop point: (sx, sy): ({measure.sample}, {measure.line})')
                x, y, dist, metrics = geom_func(base_image, measure_image,
                        bsample, bline,
                        match_func = match_func,
                        match_kwargs = match_kwargs,
                        verbose=verbose,
                        **geom_kwargs)

            except Exception as e:
                raise Exception(e)
                match_results.append(e)
                continue

            match_results.append([measure.id, measure_index, x, y,
                                 metrics, dist, base_image.file_name, measure_image.file_name])

    if verbose:
      print("Match Results: ", match_results)

    # Clean out any instances where None has been return by the geom matcher.
    match_results = np.copy(np.array([res for res in match_results if isinstance(res, list) and all(r is not None for r in res)]))

    # If all of the results are None, this point was not matchable
    if match_results.shape[0] == 0:
        raise Exception("Point with id {pointid} has no measure that matches criteria, reference measure will remain unchanged")

    # Compute the cost function for each match using the
    costs = [cost_func(match_results[:,3], match[3]) for match in match_results]

    if verbose:
      print("Values:", costs)

    # column index 3 is the metric returned by the geom matcher
    best_results = match_results[np.argmax(costs)]

    if verbose:
        print("match_results final length: ", len(match_results))
        print("best_results length: ", len(best_results))
        print("Full results: ", best_results)
        print("Winning CORRs: ", best_results[3], "Base Pixel shifts: ", best_results[4])
        print('\n')

    if len(best_results[3])==1 or best_results[3] is None:
        raise Exception("Point with id {pointid} has no measure that matches criteria, reference measure will remain unchanged")

    # Finally, update the point that will be the reference
    with ncg.session_scope() as session:
       measure = session.query(Measures).filter(Measures.id == best_results[0]).one()
       measure.sample = best_results[2]
       measure.line = best_results[3]

       point = session.query(Points).filter(Points.id == pointid).one()
       point.ref_measure = best_results[1]
    return

def estimate_logpolar_transform(img1, img2, low_sigma=0.5, high_sigma=30, verbose=False):
    """
    Estimates the rotation and scale difference for img1 that maps to img2 using phase cross correlation on a logscale projection.

    Scale and angular changes in cartesian space become translation in log-polar space. Translation from subpixel registration
    in log-polar space can then be translated into scale/rotation change in the original cartesian images. This scale + roation
    change estimation is then returned as an affine transform object. This can then be used before other subpixel registration
    methods to enable scale+rotation invariance.

    See Also
    --------

    skimage.filters.difference_of_gaussians : Bandpass filtering using a difference of gaussians
    skimage.filters.window : Simple wondowing function to remove spectral leakage along the axes in the fourier transform

    References
    ----------

    .. [1] Rittavee Matungka. 2009. Studies on log-polar transform for image registration and improvements
       using adaptive sampling and logarithmic spiral. Ph.D. Dissertation. Ohio State University, USA. Advisor(s) Yuan F. Zheng.
       Order Number: AAI3376091.
    .. [2] https://github.com/polakluk/fourier-mellin
    .. [3] https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html

    Parameters
    ----------

    img1: np.ndarray
          The source image, this is the image that is used as a base as img2 is registered to the center on img1

    img2: np.ndarray
          The image that will be moved to match img1

    low_sigma : float, list, np.array
                The low standard deviation for the Gaussian kernel used in the difference of gaussians filter. This reccomended
                to remove high frequency noise from the image before the log-polar projection as high frequency noise negatively impact registration
                in log-polar space. The lower the sigma, the sharper the resulting image will be. Use a small low_sigma with a large high_sigma
                to remove high frequency noise. Default is 0.5.

    high_sigma : float, list, np.array
                Standard deviation for the Gaussian kernel with the larger sigmas across all axes used in the difference of gaussians filter. This reccomended
                to remove high frequency noise from the image before the log-polar projection as high frequency noise negatively impact registration
                in log-polar space. The higher this sigma compared to the low_sigma, the more detail will be preserved. Use a small low_sigma with a large high_sigma
                to remove high frequency noise. A high sigma equal to ~1.6x the low sigma is reccomended for edge detection, so consider high_sigmas >= low_sigma*1.6. Default is 30.

    verbose : bool
              If true, prints out information detailing the registration process

    Returns
    -------
    : skimage.transform.SimilarityTansform
      Scikit-image affine transformation object containing rotation and scale information to warp img1 to img2

    """
    # First, band-pass filter both images
    img1 = filters.difference_of_gaussians(img1, low_sigma, high_sigma)
    img2 = filters.difference_of_gaussians(img2, low_sigma, high_sigma)

    # window images
    wimg1 = img1 * (filters.window('hann', img1.shape))
    wimg2 = img2 * (filters.window('hann', img2.shape))

    # work with shifted FFT magnitudes
    img1_fs = np.abs(fftpack.fftshift(fftpack.fft2(wimg1)))
    img2_fs = np.abs(fftpack.fftshift(fftpack.fft2(wimg2)))

    # Create log-polar transformed FFT mag images and register
    shape = img1_fs.shape
    radius = shape[0] // 4  # only take lower frequencies
    warped_img1_fs = tf.warp_polar(img1_fs, radius=radius, output_shape=shape,
                                 scaling='log', order=0)
    warped_img2_fs = tf.warp_polar(img2_fs, radius=radius, output_shape=shape,
                               scaling='log', order=0)

    warped_img1_fs = warped_img1_fs[:shape[0] // 2, :]
    warped_img2_fs = warped_img2_fs[:shape[0] // 2, :]
    shifts, error, phasediff = registration.phase_cross_correlation(warped_img1_fs,
                                                       warped_img2_fs,
                                                       upsample_factor=10)

    # Use translation parameters to calculate rotation and scaling parameters
    shiftr, shiftc = shifts[:2]
    recovered_angle = -(360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    shift_scale = np.exp(shiftc / klog)
    if recovered_angle < - 45:
        recovered_angle += 180
    else:
        if recovered_angle > 90.0:
            recovered_angle -= 180

    if verbose:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        ax = axes.ravel()
        ax[0].set_title("Original Image FFT\n(magnitude; zoomed)")
        center = np.array(shape) // 2
        ax[0].imshow(img1_fs[center[0] - radius:center[0] + radius,
                              center[1] - radius:center[1] + radius],
                     cmap='magma')
        ax[1].set_title("Modified Image FFT\n(magnitude; zoomed)")
        ax[1].imshow(img2_fs[center[0] - radius:center[0] + radius,
                            center[1] - radius:center[1] + radius],
                     cmap='magma')
        ax[2].set_title("Log-Polar-Transformed\nOriginal FFT")
        ax[2].imshow(warped_img1_fs, cmap='magma')
        ax[3].set_title("Log-Polar-Transformed\nModified FFT")
        ax[3].imshow(warped_img2_fs, cmap='magma')
        fig.suptitle('Working in frequency domain can recover rotation and scaling')
        plt.show()

        print(f"Recovered value for cc rotation: {recovered_angle}")
        print()
        print(f"Recovered value for scaling difference: {shift_scale}")

    # offset by the center of the image, scikit's ceter image rotation is defined by `axis / 2 - 0.5`
    shift_y, shift_x = np.asarray(img1.shape) / 2 - 0.5
    tf_scale = tf.SimilarityTransform(scale=shift_scale)
    tf_rotate = tf.SimilarityTransform(rotation=np.deg2rad(recovered_angle))
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

    tf_rotate_from_center = (tf_shift + (tf_rotate + tf_shift_inv))
    return tf.SimilarityTransform((tf_rotate_from_center + tf_scale)._inv_matrix)
       
def fourier_mellen(img1, img2, verbose=False, phase_kwargs={}):
    """
    Iterative phase registration using a log-polar projection to estimate an affine for scale and roation invariance.


    Parameters
    ----------

    img1: np.ndarray
          The source image, this is the image that is used as a base as img2 is registered to the center on img1

    img2: np.ndarray
          The image that will be moved to match img1

    verbose : bool
              If true, prints out information detailing the registration process

    phase_kwargs : dict
                   Parameters to be passed into the iterative_phase matcher

    Returns
    -------

    : float
      The new x coordinate for img2 that registers to the center of img1

    : float
      The new y coordinate for img2 that registers to the center of img1

    : float
      Error returned by the iterative phase matcher
    """
    # Get the affine transformation for scale + rotation invariance
    affine = estimate_logpolar_transform(img1, img2, verbose=verbose)

    # warp the source image to match the destination
    img1_warped = tf.warp(img1, affine)
    sx, sy = affine.inverse(np.asarray(img1.shape)/2)[0]

    # get translation with iterative phase
    newx, newy, error = iterative_phase(sx, sy, sx, sy, img1_warped, img2, **phase_kwargs)

    if verbose:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        ax = axes.ravel()

        ax[0].imshow(img1_warped)
        ax[0].set_title("Image 1 Transformed")
        ax[0].axhline(y=sy, color="red", linestyle="-", alpha=1, linewidth=1)
        ax[0].axvline(x=sx, color="red", linestyle="-", alpha=1, linewidth=1)

        ax[2].imshow(img1)
        ax[2].set_title("Image 1")
        ax[2].axhline(y=img1.shape[0]/2, color="red", linestyle="-", alpha=1, linewidth=1)
        ax[2].axvline(x=img1.shape[1]/2, color="red", linestyle="-", alpha=1, linewidth=1)

        ax[1].imshow(img2)
        ax[3].imshow(img2)

        if not newx or not newy:
            ax[1].set_title("Image 2 REGISTRATION FAILED")
            ax[3].set_title("Image 2 REGISTRATION FAILED")
        else :
            ax[3].set_title("Image 2 Registered")
            ax[1].axhline(y=newy, color="red", linestyle="-", alpha=1, linewidth=1)
            ax[1].axvline(x=newx, color="red", linestyle="-", alpha=1, linewidth=1)
            ax[3].axhline(y=newy, color="red", linestyle="-", alpha=1, linewidth=1)
            ax[3].axvline(x=newx, color="red", linestyle="-", alpha=1, linewidth=1)

    return newx, newy, error


# TODO: This func should be in transformation.affine with 
# signature (array_to_warp, affine, order=3).
def affine_warp_image(base_cube, input_cube, affine, order=3):
    """
    Given a base image, an input image, and an affine transformation, return
    the base image and the affine transformed input image.
    
    Parameters
    ----------
    base_cube : GeoDataset
                The base dataset that the affine transformation transforms to

    input_cube : GeoDataset
                 The cube to be transformed using the affine transformation

    affine : object
             A scikit image Affine object

    order : int
            The order of the transformation to apply. Default is a 3rd (3) order 
            polynomial.

    Returns
    -------
    base_arr : np.array()
              Original base image array

    dst_arr : np.array()
              The destination array transformed into base image's space
    """
    t1 = time.time()
    # read_array not getting correct type by default

    base_type = isis.isis2np_types[pvl.load(base_cube.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    base_arr = base_cube.read_array(dtype=base_type)

    dst_type = isis.isis2np_types[pvl.load(input_cube.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    dst_arr = input_cube.read_array(dtype=dst_type)

    box = (0, 0, max(dst_arr.shape[1], base_arr.shape[1]), max(dst_arr.shape[0], base_arr.shape[0]))
    dst_arr = np.array(Image.fromarray(dst_arr).crop(box))

    dst_arr = tf.warp(dst_arr, affine, order=order)      
    t2 = time.time()
    print(f'Affine warp took {t2-t1} seconds.')

    return base_arr, dst_arr

def subpixel_register_point_smart(pointid,
                            cost_func=lambda x,y: 1/x**2 * y,
                            ncg=None,
                            geom_func='simple',
                            match_func='classic',
                            parameters=[],
                            chooser='subpixel_register_point_smart'):

    """
    Given some point, subpixel register all of the measures in the point to the
    reference measure.

    Parameters
    ----------
    pointid : int or obj
              The identifier of the point in the DB or a Points object

    cost_func : func
                A generic cost function accepting two arguments (x,y), where x is the
                distance that a point has shifted from the original, sensor identified
                intersection, and y is the correlation coefficient coming out of the
                template matcher.

    ncg : obj
          the network candidate graph that the point is associated with; used for
          the DB session that is able to access the point.
    
    geom_func : callable
                function used to tranform the source and/or destination image before 
                running a matcher. 
    
    match_func : callable
                 subpixel matching function to use registering measures   

    parameters : list 
                 of dicts containing "match_kwargs" used for specified match_func. 
                 The passed parameters describe image and template chips that are tested.
                 For example parameters = [
                 {'match_kwargs': {'image_size':(121,121), 'template_size':(61,61)}},
                 {'match_kwargs': {'image_size':(151,151), 'template_size':(67,67)}},
                 {'match_kwargs': {'image_size':(181,181), 'template_size':(73,73)}}]
    """
    
    geom_func=geom_func.lower()
    match_func=match_func.lower()

    print(f"Using {geom_func} with the {match_func} matcher.")
    
    match_func = check_match_func(match_func)
    geom_func = check_geom_func(geom_func)

    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')
    
    if isinstance(pointid, Points):
        pointid = pointid.id
    
    t1 = time.time()
    with ncg.session_scope() as session:
        # Order by is important here because the measures get ids in sequential order when initially placed
        # and the reference_index is positionally linked to the ordered vector of measures.
        measures = session.query(Measures).filter(Measures.pointid == pointid).order_by(Measures.id).all()
        point = session.query(Points).filter(Points.id == pointid).one()
        reference_index = point.reference_index
        t2 = time.time()
        print(f'Query took {t2-t1} seconds to find the measures and reference measure.')
        
        # Get the reference measure to instantiate the source node. All other measures will
        # match to the source node.
        source = measures[reference_index]
        reference_index_id = source.imageid

        print(f'Using measure {source.id} on image {source.imageid}/{source.serial} as the reference.')
        print(f'Measure reference index is: {reference_index}')
        
        # Build a node cache so that this is an encapsulated database call. Then nodes
        # can be pulled from the lookup sans database.
        nodes = {}
        for measure in measures:
            res = session.query(Images).filter(Images.id == measure.imageid).one()
            nn = NetworkNode(node_id=measure.imageid, image_path=res.path)
            nn.parent = ncg
            nodes[measure.imageid] = nn

        session.expunge_all()

    t3 = time.time()
    print(f'Query for the image to use as source took {t3-t2} seconds.')
    print(f'Attempting to subpixel register {len(measures)-1} measures for point {pointid}')
    print(nodes)
    # Set the reference image
    source_node = nodes[reference_index_id]
    
    print(f'Source: sample: {source.sample} | line: {source.line}')
    resultlog = []
    updated_measures = []
    for i, measure in enumerate(measures):
        
        # If this is the reference node, do not attempt to match it.
        if i == reference_index:
            continue

        print()
        print(f'Measure: {measure}')
        currentlog = {'measureid':measure.id,
                    'status':''}
        cost = None

        destination_node = nodes[measure.imageid]

        print('geom_match image:', destination_node['image_path'])
        print('geom_func', geom_func)
        
        try:
            affine = estimate_affine_transformation(source_node.geodata, 
                                                        destination_node.geodata,
                                                        source.apriorisample, 
                                                        source.aprioriline)
        except Exception as e:
            print(e) 
            m = {'id': measure.id,
                 'sample':measure.apriorisample,
                 'line':measure.aprioriline,
                 'status':False,
                 'choosername':chooser}
            updated_measures.append([None, None, m])
            continue
        
        # Here is where I need to get the two ROIs extracted. Then I need to get the destination ROI affine transformed to the source ROI
        base_arr, dst_arr = affine_warp_image(source_node.geodata, 
                                              destination_node.geodata, 
                                              affine)
            
        # Compute the baseline metrics using the smallest window
        size_x = np.inf
        size_y = np.inf
        for p in parameters:
            match_kwarg = p['match_kwargs']
            if match_kwarg['template_size'][0] < size_x:
                size_x = match_kwarg['template_size'][0]
            if match_kwarg['template_size'][1] < size_y:
                size_y = match_kwarg['template_size'][1]
        
        base_roi = roi.Roi(base_arr, source.apriorisample, source.aprioriline, size_x=size_x, size_y=size_y).array
        dst_roi = roi.Roi(dst_arr, source.apriorisample, source.aprioriline, size_x=size_x, size_y=size_y).array

        if np.isnan(base_roi).any() or np.isnan(dst_roi).any():
            print('Unable to process due to NaN values in the input data.')
            m = {'id': measure.id,
                    'status': False,
                    'choosername': chooser}
            updated_measures.append([None, None, m])
            continue
        
        if base_roi.shape != dst_roi.shape:
            print('Unable to process. ROIs are different sizes for MI matcher')
            m = {'id': measure.id,
                 'status': False,
                 'choosername': chooser}
            updated_measures.append([None, None, m])
            continue

        base_roi = img_as_float32(base_roi)
        dst_roi = img_as_float32(dst_roi)

        baseline_mi = mutual_information(base_roi, dst_roi)
        

        # Refactor this call to module
        result = cv2.matchTemplate(base_roi, dst_roi, method=cv2.TM_CCOEFF_NORMED)
        baseline_corr = result[0][0]
        print(f'Baseline MI: {baseline_mi} | Baseline Corr: {baseline_corr}')
        for parameter in parameters:
            match_kwargs = parameter['match_kwargs']

            restemplate = match_func(source.apriorisample, source.aprioriline, source.apriorisample, source.aprioriline, base_arr, dst_arr, **match_kwargs)
 
            try: 
                x,y,maxcorr,temp_corrmap = restemplate
            except: 
                # did not return a corrmap 
                x,y,maxcorr = restemplate 
                temp_corrmap = np.empty((size_x, size_y))
                temp_corrmap[:] = np.nan
            
            if x is None or y is None:
                print('Unable to match with this parameter set.')
                continue
               
            base_roi = roi.Roi(base_arr, source.apriorisample, source.aprioriline, size_x=size_x, size_y=size_y).array
            dst_roi = roi.Roi(dst_arr, x, y, size_x=size_x, size_y=size_y).array

            #TODO: When refactored, all this type conversion should happen in the ROI object.
            base_roi = img_as_float32(base_roi)
            dst_roi = img_as_float32(dst_roi)

            mi_metric = mutual_information(base_roi, dst_roi)

            if mi_metric is None:
                print('MI Metric Failure. Returning.')
                m = {'id': measure.id,
                     'status': False}
            else:
                metric = maxcorr
                new_x, new_y = affine([x, y])[0]
                dist = np.linalg.norm([source.apriorisample-x, source.aprioriline-y])
                cost = cost_func(dist, metric)
    
                m = {'id': measure.id,
                    'sample':new_x,
                    'line':new_y,
                    'weight':cost,
                    'choosername':chooser,
                    'template_metric':metric,
                    'template_shift':dist, 
                    'mi_metric': mi_metric, 
                    'status': True}
                print(f'METRIC: {metric}| SAMPLE: {new_x} | LINE: {new_y} | MI: {mi_metric}')

            updated_measures.append([baseline_mi, baseline_corr, m])

    # Baseline MI, Baseline Correlation, updated measures to select from
    return updated_measures

from scipy.spatial import distance_matrix
import numpy as np

def check_for_shift_consensus(shifts, tol=0.1):
    """
    Find matched locations from a set of multiple different solutions that have 
    the same position within some user supplied tolerance. If the distance between
    two measures (shifts) is <= the tolerance, the measures are considered to have
    found consensus.

    This doc string uses 'measure' to describe each solution found by a subpixel
    matching attempt. If n-attempts are made, using n-different parameter sets,
    this function will find shift consensus between those n-different solutions.
    
    The function works by computing the full distance matrix between all solutions, 
    generating a boolean mask for distances less than the tolerance, then generating 
    a vector of column sums where the sum is the number of inliers, and finally, 
    returning a boolean vector where the column sums are greater than 2.

    Parameters
    ----------
    shifts : ndarray
             (n,2) array of (x,y) coordinates representing the subpixel registered
             measure locations. n must be >= 3.

    tol : float
          The tolerance value required for measures to be inliers. Distances between
          points less than or equal to the tolerance are inliers. In pixel space.

    Returns
     : ndarray
       (n,1) boolean array where the nth element corresponds to the nth measure
       in the shifts input array. True values indicate that the measure has shift
       consensus with at least 2 other measures
    """
    dists = distance_matrix(shifts, shifts)
    inliers = dists <= tol
    col_sums = np.sum(inliers, 1)
    # The distance matrix is zero diagonal, so 2+ means one other matcher found 
    # a close location
    return col_sums > 2
    
def decider(measures, tol=0.5):  
    """
    The logical decision function that determines which measures would be updated 
    with subpixel registration or ignored. The function iterates over the measures, 
    looks for shift consensus between subpixel registration runs.

    Parameters
    ----------
    measures : list
               A list of candidate measures (dicts) objects from the smart subpixel matcher

    tol : float
          The tolerance value required for points to be inliers. Distances between
          points less than or equal to the tolerance are inliers. In pixel space.

    Returns
    -------
    measures_to_update : list
                         of measures (dicts) to be updated to subpixel accuracy

    measures_to_set_false : list
                            of meaure ids to be ignored beause theu fail the consensus
                            building approach
    """    
    by_id = defaultdict(list)
    measures_to_set_false = []
    for m in measures:
        baseline_mi = m[0]
        baseline_corr = m[1]
        m = m[2]
        if m['status'] and m['mi_metric'] is not None and m['template_metric'] is not None:
            choosername = m['choosername']
            by_id[m['id']].append([m['line'], 
                                  m['sample'], 
                                  m['mi_metric'], 
                                  m['template_metric'], 
                                  baseline_mi, 
                                  baseline_corr, 
                                  m['template_shift']])
        else:
            measures_to_set_false.append(m['id'])
            
    measures_to_update = []
    for k, v in by_id.items():
        v = np.asarray(v)
        mi = v[:,2]
        corr = v[:,3]
        baseline_mi = v[:,4]
        baseline_corr = v[:,5]
        cost = (baseline_mi - mi) + (baseline_corr - corr)
        
        # At least two of the correlators need to have found a soln within 0.5 pixels.
        shift_mask = check_for_shift_consensus(v[:,:2], tol=tol)
        
        # This is formulated as a minimization, so the best is the min cost
        best_cost = np.argmin(cost)
        
        if shift_mask[best_cost] == False:
            # The best cost does not have positional consensus
            measures_to_set_false.append(k)
        else:
            best_measure = v[best_cost]
            m = {'id':k,
                 'line': best_measure[0],
                 'sample': best_measure[1],
                 'weight': cost[best_cost],
                 'template_metric': best_measure[3],
                 'template_shift': best_measure[6],
                 'choosername': choosername,
                 'ignore':False,
                 'best_parameter_index': best_cost}
            measures_to_update.append(m)
    # A measure could have one bad regitration and get set false, if a different parameter set passed,
    # remove from the set false list.
    ids_to_update = [d['id'] for d in measures_to_update]
    measures_to_set_false = [i for i in measures_to_set_false if i not in ids_to_update] 
    
    return measures_to_update, measures_to_set_false

def validate_candidate_measure(measure_to_register,
                            ncg=None,
                            geom_func='simple',
                            match_func='classic',
                            parameters=[],
                            **kwargs):
    """
    Compute the matching distances, matching the reference measure to the measure
    originally registered to it. This is an inverse check from the original mathcing. 
    In other words, the first registration registers A->B to find measure_to_register (B-naught).
    This func then matches B->A (B-prime) and computes the distance between B-naught and B-prime.

    Parameters
    ----------
    measure_to_register : dict
                          The measure to register

    ncg : obj
          A network candidate graph object

    geom_func : str
                The func to use to perform geometric matching

    match_func : str
                 The function to use to perform matching

    parameters : list
                 A list of matching parametrizations to test. Each entry results in
                 a subpixel registration attempt and then set of these results is
                 used ot ientify inliner and outlier parameter sets.

    Returns
    -------
    dists : list
            Of reprojection distances for each parameter set.
    """
    geom_func=geom_func.lower()
    match_func=match_func.lower()

    print(f"Using {geom_func} with the {match_func} matcher.")
    
    match_func = check_match_func(match_func)
    geom_func = check_geom_func(geom_func)

    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')
    
    measure_to_register_id = measure_to_register['id']
    
    t1 = time.time()
    with ncg.session_scope() as session:
        # Get the measure to be registered
        measure = session.query(Measures).filter(Measures.id == measure_to_register_id).order_by(Measures.id).one()
        # Get the references measure
        point = measure.point
        reference_index = point.reference_index
        reference_measure = point.measures[reference_index]

        t2 = time.time()
        print(f'Query took {t2-t1} seconds to find the measure and the reference measure.')

        # Match the reference measure to the measure_to_register - this is the inverse of the first match attempt
        # Source is the image that we are seeking to validate, destination is the reference measure. 
        # This is the inverse of other functions as this is a validator.
                
        source_imageid = measure.imageid
        source_image = session.query(Images).filter(Images.id == source_imageid).one()
        source_node = NetworkNode(node_id=source_imageid, image_path=source_image.path)
        source_node.parent = ncg
        
        destination_imageid = reference_measure.imageid
        destination_image = session.query(Images).filter(Images.id == destination_imageid).one()
        destination_node = NetworkNode(node_id=destination_imageid, image_path=destination_image.path)
        destination_node.parent = ncg
        
        sample = measure_to_register['sample']
        line = measure_to_register['line']
        
        print(f'Validating measure: {measure_to_register_id} on image: {source_image.name}')
        try:
            affine = estimate_affine_transformation(source_node.geodata, destination_node.geodata, sample, line)
        except:
            print('Unable to transform image to reference space. Likely too close to the edge of the non-reference image. Setting ignore=True')
            return [np.inf] * len(parameters)
        base_arr, dst_arr = affine_warp_image(source_node.geodata, 
                                                  destination_node.geodata, 
                                                  affine)

        dists = []
        for parameter in parameters:
            match_kwargs = parameter['match_kwargs']

            restemplate = match_func(sample, line, sample, line, base_arr, dst_arr, **match_kwargs)
 
            try: 
                x,y,maxcorr,temp_corrmap = restemplate
            except: 
                # did not return a corrmap 
                x,y,maxcorr = restemplate 
                temp_corrmap = np.empty((size_x, size_y))
                temp_corrmap[:] = np.nan
            
            if x is None or y is None:
                continue
        
        
            new_x, new_y = affine([x, y])[0]

            dist = np.sqrt((new_y - reference_measure.line) ** 2 + (new_x - reference_measure.sample) ** 2) 
            print('Reprojection Distance: ', dist)
            dists.append(dist)
        return dists

def smart_register_point(pointid, parameters=[], shared_kwargs={}, ncg=None, Session=None):    
    """
    The entry func for the smart subpixel registration code. This is the user 
    side API func for subpixel registering a point using the smart matcher.

    This function runs multiple rounds of subpixel registration on a point 
    using 'subpixel_register_point_smart', checks for a consensus from the 
    subpixel registration results, and validates the new location by inverting 
    the matching direction. This function writes to the database and outputs 
    the updated and ignored measures for logging purposes.

    This func writes to the databse. The returns are for logging and 
    debugging convenience.

    Parameters
    ----------
    pointid : int
              The id of the point to register

    parameters : list
                 A list of dict subpixel registration kwargs, {template_size: (x,x), image_size: (y,y)}

    shared_kwargs : dict
                    of kwargs passed to the subpixel matcher that are shared between all of the parameter sets

    ncg : obj
          A network candidate graph object

    Session : obj
              An optional sqlalchemy Session factory

    Returns
    -------
    measures_to_update : list
                         of measures (dicts) to be updated to subpixel accuracy

    measures_to_set_false : list
                            of meaure ids to be ignored beause theu fail the consensus
                            building approach
    
    """
    measure_results = subpixel_register_point_smart(pointid, ncg=ncg, parameters=parameters, **shared_kwargs)
    measures_to_update, measures_to_set_false = decider(measure_results)

    print()
    print(f'Found {len(measures_to_update)} measures that found subpixel registration consensus. Running validation now...')
    # Validate that the new position has consensus
    for measure in measures_to_update:
        print()
        reprojection_distances = validate_candidate_measure(measure, parameters=parameters, ncg=ncg, **shared_kwargs)
        if np.sum(np.array(reprojection_distances) < 1) < 2:
        #if reprojection_distance > 1:
            print(f"Measure {measure['id']} failed validation. Setting ignore=True for this measure.")
            measures_to_set_false.append(measure['id'])
    
    for measure in measures_to_update:
        measure['_id'] = measure.pop('id', None)

    from autocnet.io.db.model import Measures
    from sqlalchemy.sql.expression import bindparam
    
    # Update the measures that passed registration
    with ncg.engine.connect() as conn:
        if measures_to_update:
            stmt = Measures.__table__.update().\
                                    where(Measures.__table__.c.id == bindparam('_id')).\
                                    values({'weight':bindparam('weight'),
                                            'measureIgnore':bindparam('ignore'),
                                            'templateMetric':bindparam('template_metric'),
                                            'templateShift':bindparam('template_shift'),
                                            'line': bindparam('line'),
                                            'sample':bindparam('sample'),
                                            'ChooserName':bindparam('choosername')})
            resp = conn.execute(
                stmt, measures_to_update
            )
        if measures_to_set_false:
            measures_to_set_false = [{'_id':i} for i in measures_to_set_false]
            # Set ignore=True measures that failed
            stmt = Measures.__table__.update().\
                                    where(Measures.__table__.c.id == bindparam('_id')).\
                                    values({'measureIgnore':True,
                                            'ChooserName':shared_kwargs['chooser']})
            resp = conn.execute(
                stmt, measures_to_set_false
            )
    return measures_to_update, measures_to_set_false
