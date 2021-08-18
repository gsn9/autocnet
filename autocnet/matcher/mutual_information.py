from math import floor

import numpy as np

from scipy.ndimage.measurements import center_of_mass

def mutual_information(t1, t2, **kwargs):
    """
    Computes the correlation coefficient between two images using a histogram
    comparison (Mutual information for joint histograms). The corr_map coefficient
    will be between 0 and 4

    Parameters
    ----------

    t1 : ndarray
         First image to use in the histogram comparison

    t2 : ndarray
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
    hgram, x_edges, y_edges = np.histogram2d(t1.ravel(),t2.ravel(), **kwargs)

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

    bins : int
           Number of bins to use when computing the histograms

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

    image_size = s_image.shape
    template_size = d_template.shape

    y_diff = image_size[0] - template_size[0]
    x_diff = image_size[1] - template_size[1]

    max_corr = -np.inf
    corr_map = np.zeros((y_diff+1, x_diff+1))
    max_i = -1  # y
    max_j = -1  # x
    for i in range(y_diff+1):
        for j in range(x_diff+1):
            sub_image = s_image[i:i+template_size[1],  # y
                                j:j+template_size[0]]  # x
            corr = func(sub_image, d_template, **kwargs)
            if corr > max_corr:
                max_corr = corr
                max_i = i
                max_j = j
            corr_map[i, j] = corr

    y, x = np.unravel_index(np.argmax(corr_map, axis=None), corr_map.shape)

    upper = int(2 + floor(subpixel_size / 2))
    lower = upper - 1
    # x, y are the location of the upper left hand corner of the template in the image
    area = corr_map[y-lower:y+upper,
                    x-lower:x+upper]

    # Compute the y, x shift (subpixel) using scipys center_of_mass function
    cmass  = center_of_mass(area)

    if area.shape != (subpixel_size+2, subpixel_size+2):
        print("Max correlation is too close to the boundary.")
        return None, None, 0, None

    subpixel_y_shift = subpixel_size - 1 - cmass[0]
    subpixel_x_shift = subpixel_size - 1 - cmass[1]

    y += subpixel_y_shift
    x += subpixel_x_shift

    # Compute the idealized shift (image center)
    y -= (s_image.shape[0] / 2) - (d_template.shape[0] / 2)
    x -= (s_image.shape[1] / 2) - (d_template.shape[1] / 2)

    return float(x), float(y), float(max_corr), corr_map
