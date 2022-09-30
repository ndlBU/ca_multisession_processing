#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:52:20 2020

@author: jad
"""

import logging
import numpy as np
from scipy.sparse import csc_matrix
import scipy.ndimage as ndimage


def get_spatial_mask(A, dims, thr=0.9):
    """
    Gets mask of spatial components corresponding to the pixels that contribute
    thr fraction of energy.

    Parameters
    ----------
    A : np.ndarray or sparse matrix of Spatial components [d x K]
                   d: num pixels (width x height)
                   K: number of components/neurons
    dims : (int, int)
        dimensions (height, width) of mask.
    thr: scalar between 0 and 1
                   Energy threshold for computing contours (default 0.9)
                   Keeps the pixels that contribute up to thr of the energy
    Returns
    -------
    M : bool sparse matrix [d x K]
            Masks: Boolean matrix corresponding to thr fraction of spatial component energy.

    """
    if 'csc_matrix' not in str(type(A)):
        A = csc_matrix(A)
    d, nr = np.shape(A)
    
    M = A.copy()

    # for each patches
    for i in range(nr):
        # we compute the cumulative sum of the energy of the Ath component that has been ordered from highest to lowest
        patch_data = A.data[A.indptr[i]:A.indptr[i + 1]]
        indx = np.argsort(patch_data)[::-1]
        cum_en = np.cumsum(patch_data[indx]**2)
        
        if len(cum_en) == 0:
            continue
        elif len(cum_en) < 10:
            cum_en /= cum_en[-1]
        else:
            # we work with normalized values
            cum_en /= cum_en[-1]
            cum_en[cum_en > thr] = 0

        patch_data_updated = np.empty_like(patch_data)
        patch_data_updated[indx] = cum_en

        M.data[A.indptr[i]:A.indptr[i + 1]] = patch_data_updated

    M.data = M.data.astype('bool')
    return M


# center of mass
def get_centroids(A:np.ndarray, d1: int, d2: int, order='f'):
    """
    Calculation of the center of mass for spatial components
     Args:
         A:   np.ndarray
              matrix of spatial components (d x K) where d = d1 x d2
         d1:  int
              number of pixels in 1st direction
         d2:  int
              number of pixels in 2nd direction
         order: 'c' or 'f'
              use c array order (ascending index) or fortran order (descending)
     Returns:
         centroids:  np.ndarray
              center of mass for spatial components (K x 2 or 3)
    """

    if 'csc_matrix' in str(type(A)):
        A = A.toarray()

    coord = np.array([np.outer(np.ones(d2), np.arange(d1)).ravel(),
                      np.outer(np.arange(d2), np.ones(d1)).ravel()],
                     dtype=A.dtype)

    centroids = np.array((coord @ A / A.sum(axis=0)).T)

    if order == 'f':
        return centroids
    elif order == 'c':
        return centroids[:, ::-1]


def smooth_roi(A, dims=(256, 256), dilation_kernel=None, erosion_kernel=None,
               median_filter_size=5):
    """
    Peforms a binary closing operation (dilation followed by erosion) using the 
    dilation and erosion kernels. The output denoised using a median filter to
    close any missed gaps and remove stray pixels.

    Parameters
    ----------
    A : csc_matrix [d x nElements], (d = dims[0]*dims[1])
        column sparse matrix of spatial footprints. Each column is a component
    dims : tuple (int, int)
        dimension (height, width) of field of view.
    dilation_kernel : ndarray , optional
        Smoothing kernel used int the dilation operation. If not provided, a circular
        kernel with diameter median filter kernel size is used. The default is None.
    erosion_kernel : ndarray , optional
        Smoothing kernel used int the erosion operation. If not provided, a circular
        kernel with diameter median filter kernel size is used. The default is None.
    median_filter_size : int, optional
        Kernel size (widht and height) of median filter. The default is 5.

    Returns
    -------
    csc_matrix [d x nElements], (d = d1*d2)
        Smoothed spatial matrix.

    """
    if 'csc_matrix' not in str(type(A)):
        A = csc_matrix(A)
        
    if dilation_kernel is None:
        dilation_kernel = create_circular_mask(median_filter_size, median_filter_size)
    
    if erosion_kernel is None:
        erosion_kernel = create_circular_mask(median_filter_size, median_filter_size)
    
    Asmooth=[]
    for mask in A.T:
        Atmp = mask.toarray().reshape(dims, order='F')
        # Dilate with kernel to fill out small gaps, connect areas and smooth outline
        Atmp= ndimage.binary_dilation(Atmp, structure=dilation_kernel)
        # Erode with kernel to retain the original size
        Atmp= ndimage.binary_erosion(Atmp, structure=erosion_kernel)
        # Median filter to remove salt and pepper noise
        Atmp = ndimage.median_filter(Atmp, size=median_filter_size)

        Atmp = Atmp.flatten(order='F')
        Asmooth.append(Atmp)

    Asmooth = np.array(Asmooth)
    return csc_matrix(Asmooth.T).astype('int')


def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates a circular mask. Assumes image indexing (Y,X)
    
    Parameters
    ----------
    h : int
        Height of mask.
    w : int
        Width of mask.
    center : tuple (Y,X), optional
        Coordinates of center. The default is None.
    radius : int, optional
        Radius of mask. The default is None.

    Returns
    -------
    ndarray int
        Mask warray.
    """

    if center is None:  # use the middle of the image
        center = (int(h/2), int(w/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h-center[0], w-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)

    mask = dist_from_center <= radius
    return mask.astype('int')


def dfs(A, dims, x, y):
    """
    Recursive depth first search of a graph to travese the ROI. Locations passed
    are set to 0.
    NOTE: sets the values to 0 in the provided ROI reference.
    
    Parameters
    ----------
    A : ndarray [height x width] (dims[0] x dims[1])
        Array of spatial footprint.
    dims : tuple (int, int)
        dimension (height, width) of field of view.
    x : int
        current x-coordinate in the ROI.
    y : int
        current y-coordinate in the ROI.

    Returns
    -------
    None.
    """
    d0, d1 = dims
    if A[y, x] == 0:
        return
    A[y, x] = 0

    if y != 0:
        dfs(A, dims, x, y-1)
    if y != d0-1:
        dfs(A, dims, x, y+1)    
    if x != 0:
        dfs(A, dims, x-1, y)
    if x != d1-1:
        dfs(A, dims, x+1, y)


def dfs_label(A, A_label, dims, x, y, label):
    """
    Recursive depth first search of a graph to travese the ROI. Locations passed
    are set to 0. Sets all points in label matrix at the coordinates of traversal
    to the provided label.
    NOTE: sets the values to 0 in the provided ROI reference.

    Parameters
    ----------
    A : ndarray [height x width] (dims[0] x dims[1])
        Array of spatial footprint.
    A_label : ndarray [dims[0] x dims[1]]
        Label array that is filled in place.
    dims : tuple (int, int)
        dimension ( height, width) of field of view.
    x : int
        current x-coordinate in the ROI.
    y : int
        current y-coordinate in the ROI.
    label : int
        label for the connected fragment.

    Returns
    -------
    None.
    """
    d0, d1 = dims
    if A[y, x] == 0:
        return
    A[y, x] = 0
    A_label[y, x] = label

    if y != 0:
        dfs_label(A, A_label, dims, x, y - 1, label)
    if y != d0-1:
        dfs_label(A, A_label, dims, x, y + 1, label)
    if x != 0:
        dfs_label(A, A_label, dims, x - 1, y, label)
    if x != d1-1:
        dfs_label(A, A_label, dims, x + 1, y, label)
    

def label_fragment(A, dims):
    """
    Labels the unique fragments (4-neighbor islands) in ROI. Uses a 
    recursive depth first search of a graph to travese the ROI.
    
    NOTE: sets the values to 0 in the provided ROI reference. Provide a copy to
    this function to avoid overwriting the original data.
    
    Parameters
    ----------
    A : ndarray [height x width] (dims[0] x dims[1])
        Array of spatial footprint.
    dims : tuple (int, int)
        dimension (height, width) of field of view.

    Returns
    -------
    A_label : ndarray [dims[1] x dims[0]]
        Label array with fragments having unique integer labels.
    """
    d0, d1 = dims
    labelA = np.empty(dims)
    labelA[:] = 0
    label = 0
    for y in range(d0):
        for x in range(d1):
            if A[y,x] == 1:
                label += 1
                dfs_label(A, labelA, dims, x, y, label)
    return labelA


def count_fragments(A, dims):
    """
    Counts the number of unique fragments (4-neighbor islands) in ROI. Uses a 
    recursive depth first search of a graph to travese the ROI.
    
    NOTE: sets the values to 0 in the provided ROI reference. Provide a copy to
    this function to avoid overwriting the original data.
    
    Parameters
    ----------
    A : ndarray [height x width] (dims[0] x dims[1])
        Array of spatial footprint.
    dims : tuple (int, int)
        dimension (height, width) of field of view.

    Returns
    -------
    count : int
        Number of unique fragments in ROI.
    """
    d0, d1 = dims
    count = 0
    for y in range(d0):
        for x in range(d1):
            if A[y,x] == 1:
                dfs(A, dims, x, y)
                count +=1
    return count


def defragment_roi(Ain, dims, fragment_count, thr_size=0.75):
    """
    For fragmented ROIs, if a fragment is thr_size of the total spatial extent, it
    is kept as the component. Otherwise the component is flagged as unusable in
    future processing steps.
    
    Parameters
    ----------
    Ain : ndarray [Y x X] (dims[0] x dims[1])
        Array of spatial footprint.
    dims : tuple (int, int)
        dimension (height, height) of field of view.
    fragment_count : int
        Number of identified fragments.
    thr_size : calar , optional
        Threshold in [0,1] for ratio of largest fragment to footprint so that
        fragment is kept as component. The default is 0.75.

    Returns
    -------
    Aout : ndarray [Y x X]
        Array of spatial footprint.
    keep_roi : bool
        Indicated whether an element should be used further. Is false for fragmented elements
    labeledA : ndarray [Y x X]
         Array of spatial footprints with each fragment having a unique int label.

    """
    
    Aout = np.zeros_like(Ain)
    keep_roi = False
    nels = np.sum(Ain).astype('float')
    labeledA = label_fragment(Ain, dims)

    # chech if any fragment has area greater than threshold.
    for j in range(fragment_count):
        if np.sum(labeledA == j+1)/nels >= thr_size:
            keep_roi = True
            # Adjust ROI to be this fragment only
            Aout[labeledA == j+1] = 1
            break
    # return orginal if no frament large enough
    if not keep_roi:
        Aout[labeledA > 0] = 1

    return Aout, keep_roi, labeledA


def cleanup_rois(Ain, dims=(256, 256), dilation_kernel=None, erosion_kernel=None,
                 median_filter_size=5, fragment_thr=0.75):
    """
    Each ROI undergoes a smoothing operation composed of a closing operation defined
    by the dilation and erosion kernels, followed by median filtering.
    
    For fragmented ROIs, if a fragment is fragthr of the total spatial extent, it
    is kept as the component. Otherwise the component is flagged as unusable in
    future processing steps.
    

    Parameters
    ----------

    Ain :  csc_matrix or ndarray [d x nElements], (d = dims[0]*dims[1])
        column sparse matrix of spatial footprints. Each column is a component
    dims : tuple (int, int)
        dimension (height, width) of field of view.
    dilation_kernel : ndarray (or int), optional
        Smoothing kernel used int the dilation operation. If integer is provided, 
        it is used as diameter of circular kernel. If not provided, a circular
        kernel with radius median filter kernel size is used. The default is None.
    erosion_kernel : ndarray (or int) , optional
        Smoothing kernel used int the erosion operation. If integer is provided, 
        it is used as diameter of circular kernel. If not provided, a circular
        kernel with radius median filter kernel size is used. The default is None.
    median_filter_size : int, optional
        Kernel size (widht and height) of median filter. The default is 5.
    fragment_thr : scalar , optional
        Threshold in [0,1] for ratio of largest fragment to footprint so that
        fragment is kept as component. The default is 0.75.
        
    Returns
    -------
    Aout : csc_matrix [d x nElements], (d = dims[0]*dims[1])
        column sparse matrix of processed spatial footprints. Each column is a component.
    keep_roi : bool ndarray [nElements]
        Indicated whether an element should be used further. Is false for fragmented elements
    bad_rois : csc_matrix [d x nFragmentedElements], (d = dims[0]*dims[1])
        column sparse matrix of spatial footprints for fragmented componets.
        Each fragment has a unique int label.

    """
    if 'csc_matrix' not in str(type(Ain)):
        Ain = csc_matrix(Ain)
    
    if dilation_kernel is None:
        dilation_kernel = create_circular_mask(median_filter_size, median_filter_size)
    elif 'int' in str(type(dilation_kernel)):
        dilation_kernel = create_circular_mask(dilation_kernel, dilation_kernel)
    elif 'ndarray' in str(type(dilation_kernel)) and dilation_kernel.ndim == 2:
        pass
    else:
        raise ValueError("Dilation kernel should be 2D array or scalar for generating circular mask")

    if erosion_kernel is None:
        erosion_kernel = create_circular_mask(median_filter_size, median_filter_size)
    elif 'int' in str(type(erosion_kernel)):
        erosion_kernel = create_circular_mask(erosion_kernel, erosion_kernel)
    elif 'ndarray' in str(type(erosion_kernel)) and erosion_kernel.ndim == 2:
        pass
    else:
        raise ValueError("Erosion kernel should be 2D array or scalar for generating circular mask")

    Aout = []
    keep_roi = []
    bad_rois = []

    # smooth the footprints with closing followed by median filtering
    A_smoothedROI = smooth_roi(Ain, dims=dims, dilation_kernel=dilation_kernel, erosion_kernel=erosion_kernel, median_filter_size=median_filter_size)
    A_smoothedROI = A_smoothedROI.toarray()
    
    for i, mask in enumerate(A_smoothedROI.T):
        masktmp = mask.copy().reshape(dims, order='F')
        cnt =count_fragments(masktmp, dims)
        if cnt == 1:
            # keep monolithic footprints as they are
            keep_roi.append(True)
            Aout.append(mask)
        elif cnt > 1:
            logging.debug('Checking component ', str(i), ' with fragments ', str(cnt))
            masktmp = mask.copy().reshape(dims, order='F')
            # check if ROI can be defragmented
            maskDefrag, keepComponent, maskLabeled = defragment_roi(masktmp, dims, cnt, thr_size=fragment_thr)
            Aout.append(maskDefrag.flatten(order='F'))
            keep_roi.append(keepComponent)
            # keep labeled bad ROI if it cannot be defragmented
            if not keepComponent:
                bad_rois.append(maskLabeled.flatten(order='F'))
        else:
            Aout.append(mask)
            keep_roi.append(False)

    # adjust types and return
    Aout = csc_matrix(np.array(Aout).T).astype('int')
    bad_rois = csc_matrix(np.array(bad_rois).T)
    keep_roi = np.array(keep_roi)

    return Aout, keep_roi, bad_rois


def shift_image(img, offset=(0, 0), fill=None):
    """
    Shifts the image and fills in values past the input edges.
    Parameters
    ----------
    img : ndarray [ndim=2]
        Image to be shifted.
    offset : tuple (int, int), optional
        The shift (dY, dX) applied to the input image. The default is (0,0).
    fill : str or scalar, optional
        Value to fill past edges of input in numeric.
        If str in {'mean', 'median'} uses the specified statistic to fill.
        The default is None.

    Returns
    -------
    img_out : ndarray [ndim=2]
        Image translated by offset.
    """

    if np.size(offset) != 2:
        raise ValueError('The offset should be provided for 2 dimensions')
    
    if fill is None:
        f = 0
    elif fill == 'mean':
        f = np.mean(img)
    elif fill == 'median':
        f = np.median(img)
    elif isinstance(fill, int) or isinstance(fill, float):
        f = fill
    else:
        f = 0
    
    img_out = ndimage.shift(img, offset, cval=f)

    return img_out
