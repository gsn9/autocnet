"""This module contains wrappers around the ISIS campt and mappt that are
specific to understanding the relationship between pixels and projected
coordinates."""

# This is free and unencumbered software released into the public domain.
#
# The authors of autocnet do not claim copyright on the contents of this file.
# For more details about the LICENSE terms and the AUTHORS, you will
# find files of those names at the top level of this repository.
#
# SPDX-License-Identifier: CC0-1.0

import os
from collections import abc
from numbers import Number

import numpy as np

try:
    import kalasiris as isis
except Exception as exception:
    from autocnet.utils.utils import FailedImport
    isis = FailedImport(exception)

import pvl

isis2np_types = {
        "UnsignedByte" : "uint8",
        "SignedWord" : "int16",
        "Double" : "float64",
        "Real" : "float32"
}

np2isis_types = {v: k for k, v in isis2np_types.items()}


def get_isis_special_pixels(arr):
    """
    Returns coordinates of any ISIS no data pixels. Essentially, 
    np.argwhere results of where pixels match ISIS special 
    data types (NIRs, NHRs, HIS, HRS, NULLS).

    Parameters
    ----------
    arr : np.array 
          Array to find special pixels in 
    
    Returns
    -------
    : sp
      np.array of coordinates in y,x format containing special pixel coordinates

    """
    isis_dtype = np2isis_types[str(arr.dtype)]
    sp_pixels = getattr(isis.specialpixels, isis_dtype)

    null = np.argwhere(arr==sp_pixels.Null)
    lrs = np.argwhere(arr==sp_pixels.Lrs)
    lis = np.argwhere(arr==sp_pixels.Lis)
    his = np.argwhere(arr==sp_pixels.His)
    hrs = np.argwhere(arr==sp_pixels.Hrs)
    sp = np.concatenate((null, lrs, lis, his, hrs))

    return sp


def get_nodata_bounds(arr):
    """
    Get bounds for an image that does not contain any ISIS special pixels. That is,
    ISIS Nulls, NIRS, NRS, HIS and HRS pixels

    Parameters
    ----------
    arr : np.array
          2D array representing the image 

    Returns
    -------
    : left_x 
      left x coordinate of new bounds 

    : right_x
      right x coordinate of new bounds 
    
    : top_y
      top y coordinate of new bounds 
    
    : bottom _y
      bottom y coordinates of new bounds 
    """
    sp = get_isis_special_pixels(arr)
    
    if not sp.any():
        return 0, arr.shape[1], 0, arr.shape[0]
    
    cy, cx = arr.shape[1]//2, arr.shape[0]//2
    tree = KDTree(sp, metric='euclidean')

    # For finding K neighbors of P1 with shape (1, 3)
    distances, indices = tree.query(np.array([cy, cx]).reshape(1,2), 1)
    
    # these are slightly misshapen by being in nested arrays (e.g. [[n]], [[y,x]])
    nearest_idx = indices.reshape(1,)
    neary, nearx = sp[nearest_idx].reshape(2,)

    # subtract 1 to exclude the special pixel
    x_dist = abs(cx - nearx) - 1
    y_dist = abs(cy - neary) - 1

    # left_x, right_x, top_y, bottom_y
    left_x = cx - x_dist
    right_x = cx + x_dist
    top_y = cy - y_dist
    bottom_y = cy + y_dist

    return left_x, right_x, top_y, bottom_y


def point_info(
        cube_path: os.PathLike,
        x,
        y,
        point_type: str,
        allowoutside=False
):
    """
    Returns a pvl.collections.MutableMappingSequence object or a
    Sequence of MutableMappingSequence objects which contain keys
    and values derived from the output of ISIS campt or mappt on
    the *cube_path*.

    If x and y are single numbers, then a single MutableMappingSequence
    object will be returned.  If they are Sequences or Numpy arrays, then a
    Sequence of MutableMappingSequence objects will be returned,
    such that the first MutableMappingSequence object of the returned
    Sequence will correspond to the result of *x[0]* and *y[0]*,
    etc.

    Raises subprocess.CalledProcessError if campt or mappt have failures.
    May raise ValueError if campt completes, but reports errors.

    Parameters
    ----------
    cube_path : os.PathLike
                Path to the input cube.

    x : Number, Sequence of Numbers, or Numpy Array
        Point(s) in the x direction. Interpreted as either a sample
        or a longitude value determined by *point_type*.

    y : Number, Sequence of Numbers, or Numpy Array
        Point(s) in the y direction. Interpreted as either a line
        or a latitude value determined by *point_type*.

    point_type : str
                 Options: {"image", "ground"}
                 Pass "image" if  x,y are in image space (sample, line) or
                 "ground" if in ground space (longitude, latitude)

    allowoutside: bool
                  Defaults to False, this parameter is passed to campt
                  or mappt.  Please read the ISIS documentation to
                  learn more about this parameter.

    """
    point_type = point_type.casefold()
    valid_types = {"image", "ground"}
    if point_type not in valid_types:
        raise ValueError(
            f'{point_type} is not a valid point type, valid types are '
            f'{valid_types}'
        )

    if isinstance(x, abc.Sequence) and isinstance(y, abc.Sequence):
        if len(x) != len(y):
            raise IndexError(
                f"Sequences given to x and y must be of the same length."
            )
        x_coords = x
        y_coords = y
    elif isinstance(x, Number) and isinstance(y, Number):
        x_coords = [x, ]
        y_coords = [y, ]
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if not all((x.ndim == 1, y.ndim == 1)):
            raise IndexError(
                f"If they are numpy arrays, x and y must be one-dimensional, "
                f"they were: {x.ndim} and {y.ndim}"
            )
        if x.shape != y.shape:
            raise IndexError(
                f"Numpy arrays given to x and y must be of the same shape."
            )
        x_coords = x
        y_coords = y
    else:
        raise TypeError(
            f"The values of x and y were neither Sequences nor individual "
            f"numbers, they were: {x} and {y}"
        )

    results = []
    if pvl.load(cube_path).get("IsisCube").get("Mapping"):
        # We have a projected image, and must use mappt
        mappt_common_args = dict(allowoutside=allowoutside, type=point_type)

        for xx, yy in zip(x_coords, y_coords):
            mappt_args = {
                "ground": dict(
                    longitude=xx,
                    latitude=yy,
                    coordsys="UNIVERSAL"
                ),
                "image": dict(
                    # Convert PLIO pixels to ISIS pixels
                    sample=xx+0.5,
                    line=yy+0.5
                )
            }
            for k in mappt_args.keys():
                mappt_args[k].update(mappt_common_args)
            mapres = pvl.loads(isis.mappt(cube_path, **mappt_args[point_type]).stdout)["Results"]
            
            # convert from ISIS pixels to PLIO pixels
            mapres['Sample'] = mapres['Sample'] - 0.5
            mapres['Line'] = mapres['Line'] - 0.5

            results.append(mapres)
    else:
        # Not projected, use campt
        if point_type == "ground":
            # campt uses lat, lon for ground but sample, line for image.
            # So swap x,y for ground-to-image calls
            p_list = [f"{lat}, {lon}" for lon, lat in zip(x_coords, y_coords)]
        else:
            p_list = [
                f"{samp+0.5}, {line+0.5}" for samp, line in zip(x_coords, y_coords)
            ]

        # ISIS's campt needs points in a file
        with isis.fromlist.temp(p_list) as f:
            cp = isis.campt(
                cube_path,
                coordlist=f,
                allowoutside=allowoutside,
                usecoordlist=True,
                coordtype=point_type
            )

        camres = pvl.loads(cp.stdout)
        for r in camres.getall("GroundPoint"):
            if r['Error'] is None:
                # convert all pixels to PLIO pixels from ISIS
                r["Sample"] -= .5
                r["Line"] -= .5
                results.append(r)
            else:
                raise ValueError(
                    f"ISIS campt completed, but reported an error: {r['Error']}"
                )

    if isinstance(x, (abc.Sequence, np.ndarray)):
        return results
    else:
        return results[0]


def image_to_ground(
        cube_path: os.PathLike,
        sample,
        line,
        lontype="PositiveEast360Longitude",
        lattype="PlanetocentricLatitude",
):
    """
    Returns a two-tuple of numpy arrays or a two-tuple of floats, where
    the first element of the tuple is the longitude(s) and the second
    element are the latitude(s) that represent the coordinate(s) of the
    input *sample* and *line* in *cube_path*.

    If *sample* and *line* are single numbers, then the returned two-tuple
    will have single elements. If they are Sequences, then the returned
    two-tuple will contain numpy arrays.

    Raises the same exceptions as point_info().

    Parameters
    ----------
    cube_path : os.PathLike
                Path to the input cube.

    sample : Number or Sequence of Numbers
        Sample coordinate(s).

    line : Number or Sequence of Numbers
        Line coordinate(s).

    lontype: str
        Name of key to query in the campt or mappt return to get the returned
        longitudes. Defaults to "PositiveEast360Longitude", but other values
        are possible. Please see the campt or mappt documentation.

    lattype: str
        Name of key to query in the campt or mappt return to get the returned
        latitudes. Defaults to "PlanetocentricLatitude", but other values
        are possible.  Please see the campt or mappt documentation.

    """
    res = point_info(cube_path, sample, line, "image")

    if isinstance(sample, (abc.Sequence, np.ndarray)):
        lon_list = list()
        lat_list = list()
        for r in res:
            lon_list.append(_get_value(r[lontype]))
            lat_list.append(_get_value(r[lattype]))

        lons = np.asarray(lon_list)
        lats = np.asarray(lat_list)
    else:
        lons = _get_value(res[lontype])
        lats = _get_value(res[lattype])

    return lons, lats


def _get_value(obj):
    """Returns *obj*, unless *obj* is of type pvl.collections.Quantity, in
    which case, the .value component of the object is returned."""
    if isinstance(obj, pvl.collections.Quantity):
        return obj.value
    else:
        return obj


def ground_to_image(cube_path, lon, lat):
    """
    Returns a two-tuple of numpy arrays or a two-tuple of floats, where
    the first element of the tuple is the sample(s) and the second
    element are the lines(s) that represent the coordinate(s) of the
    input *lon* and *lat* in *cube_path*.

    If *lon* and *lat* are single numbers, then the returned two-tuple
    will have single elements. If they are Sequences, then the returned
    two-tuple will contain numpy arrays.

    Raises the same exceptions as point_info().

    Parameters
    ----------
    cube_path : os.PathLike
                Path to the input cube.

    lon: Number or Sequence of Numbers
        Longitude coordinate(s).

    lat: Number or Sequence of Numbers
        Latitude coordinate(s).

    """
    res = point_info(cube_path, lon, lat, "ground")

    if isinstance(lon, (abc.Sequence, np.ndarray)):
        samples, lines = np.asarray([[r["Sample"], r["Line"]] for r in res]).T
    else:
        samples, lines = res["Sample"], res["Line"]

    return samples, lines


