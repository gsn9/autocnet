import numpy as np

def mi(t1, t2, **kwargs):
    """
    Computes the correlation coefficient between two images using a histogram
    comparison (Mutual information for joint histograms). The result coefficient
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

def mutual_information(d_template, s_image, bins=100, func=mi):
    """
    Applys the mutual information matcher function over a search image using a
    defined template. Where the search area is 2x the size of the template image


    Parameters
    ----------
    template : ndarray
               The input search template used to 'query' the destination
               image

    image : ndarray
            The image or sub-image to be searched

    bins : int
           Number of bins to use when computing the histograms

    Returns
    -------
    x : float
        The x offset

    y : float
        The y offset

    max_corr : float
               The strength of the correlation in the range [-1, 1].

    corr_map : ndarray
               Map of corrilation coefficients when comparing the template to
               locations within the search area
    """
    image_size = s_image.shape
    template_size = d_template.shape

    y_diff = abs(template_size[0] - image_size[0])
    x_diff = abs(template_size[1] - image_size[1])

    max_corr = -np.inf
    corr_map = np.zeros((y_diff+1, x_diff+1))
    max_i = -1  # y
    max_j = -1  # x
    for i in range(y_diff+1):
        for j in range(x_diff+1):
            sub_image = s_image[i:i+template_size[1],  # y
                                j:j+template_size[0]]  # x
            corr = func(sub_image, d_template, bins=bins)
            if corr > max_corr:
                max_corr = corr
                max_i = i
                max_j = j
            corr_map[i, j] = corr

    # This is still operating at the pixel scale. Use the template_match_autoreg
    # logic to achieve submpixel weighting.
    x_offset = max_j - (template_size[1]/2)
    y_offset = max_i - (template_size[0]/2)

    return x_offset, y_offset, max_corr, corr_map
