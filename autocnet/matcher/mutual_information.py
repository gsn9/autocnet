from math import floor
from autocnet.transformation.roi import Roi
import numpy as np

from scipy.ndimage.measurements import center_of_mass
from skimage.transform import AffineTransform

def mutual_information(reference_roi, moving_roi, affine=AffineTransform(), **kwargs):
    """
    Computes the correlation coefficient between two images using a histogram
    comparison (Mutual information for joint histograms). The corr_map coefficient
    will be between 0 and 4

    Parameters
    ----------

    reference_roi : Roi
                    First image to use in the histogram comparison
    
    moving_roi : Roi
                   Second image to use in the histogram comparison
    
    
    Returns
    -------

    : float
      Correlation coefficient computed between the two images being compared
      between 0 and 4

    See Also
    --------
    numpy.histogram2d : for the kwargs that can be passed to the comparison
    """
    
    # grab ndarray from input Roi's 
    reference_image = reference_roi.clip()
    walking_template = moving_roi.clip()
    
    # if reference_roi.ndv == None or moving_roi.ndv == None:
    if np.isnan(reference_image).any() or np.isnan(walking_template).any():
        raise Exception('Unable to process due to NaN values in the input data')
    
    if reference_roi.size_y != moving_roi.size_y and reference_roi.size_x != moving_roi.size_x:
    # if reference_image.shape != moving_roi.shape:
        raise Exception('Unable compute MI. Image sizes are not identical.')

    hgram, x_edges, y_edges = np.histogram2d(reference_image.ravel(), walking_template.ravel(), **kwargs)

    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def mutual_information_match(d_template, s_image, subpixel_size=3,
                             func=None, **kwargs):
    """
    Applys the mutual information matcher function over a search image using a
    defined template


    Parameters
    ----------
    d_template : ndarray
                 The input search template used to 'query' the destination
                 image

    s_image : ndarray
              The image or sub-image to be searched

    subpixel_size : int
                    Subpixel area size to search for the center of mass
                    calculation

    func : function
           Function object to be used to compute the histogram comparison

    Returns
    -------
    x : float
        The x offset

    y : float
        The y offset

    max_corr : float
               The strength of the correlation in the range [0, 4].

    corr_map : ndarray
               Map of corrilation coefficients when comparing the template to
               locations within the search area
    """

    if func == None:
        func = mutual_information


    image_size = (s_image.size_x * 2, s_image.size_y * 2)
    template_size = (d_template.size_x * 2, d_template.size_y * 2)

    y_diff = image_size[0] - template_size[0]
    x_diff = image_size[1] - template_size[1]

    max_corr = -np.inf
    corr_map = np.zeros(template_size)
    max_i = -1  # y
    max_j = -1  # x

    s_image_extent = s_image.image_extent

    starting_y = s_image_extent[2]
    ending_y = s_image_extent[3]
    starting_x = s_image_extent[0]
    ending_x = s_image_extent[1]

    for i in range(starting_y, ending_y):
        for j in range(starting_x, ending_x):

            s_image.x = (j)
            s_image.y = (i)
           
            corr = func(s_image, d_template, **kwargs)
            if corr > max_corr:
                max_corr = corr
                max_i = i - s_image_extent[2]
                max_j = j - s_image_extent[0]
            

            corr_map[i- s_image_extent[2], j - s_image_extent[0]] = corr

    y, x = np.unravel_index(np.argmax(corr_map, axis=None), corr_map.shape)

    upper = int(2 + floor(subpixel_size / 2))
    lower = upper - 1

    area = corr_map[y-lower:y+upper,
                    x-lower:x+upper]

    # Compute the y, x shift (subpixel) using scipys center_of_mass function
    cmass  = center_of_mass(area)

    if area.shape != (subpixel_size+2, subpixel_size+2):
        return None, None, 0, None

    subpixel_y_shift = subpixel_size - 1 - cmass[0]
    subpixel_x_shift = subpixel_size - 1 - cmass[1]
    y = abs(y - (corr_map.shape[1])/2)
    x = abs(x - (corr_map.shape[0])/2)
    y += subpixel_y_shift
    x += subpixel_x_shift
    new_affine = AffineTransform(translation=(-x, -y))
    return new_affine, float(max_corr), corr_map
    # return float(x), float(y), float(max_corr), corr_map