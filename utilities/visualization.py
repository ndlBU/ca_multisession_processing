#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Created on 12-Sep-2022 at 1:25 PM

# @author: jad
"""

from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.color import label2rgb
from skimage.measure import find_contours
from scipy.sparse import csc_matrix

from utilities.utilities import old_div
from utilities.fluo_processing import get_spatial_mask, get_centroids

DEFAULT_COLORS = sns.color_palette('muted')[-1:] + sns.color_palette('muted')[0:-1]


def label_regions(rois_in, img_bck, dims, alpha, thr, colors=None):
    # Create the labeled image overlay
    images_masked = []
    for i, A in enumerate(rois_in):
        images_masked.append(
            (i+1) * get_spatial_mask(A, dims, thr=thr).toarray().any(axis=1).reshape(dims, order='F').astype(int))

    images_masked = np.array(images_masked).max(axis=0)
    img = label2rgb(images_masked, image=img_bck, bg_label=0, colors=colors)
    return img


def plot_roi_masked(rois, dims, img_in=None, color_list=DEFAULT_COLORS, start_index=0,
                    display_numbers=False, thr=0.9, alpha=0.2, axes_off=True, figure_size=(8, 8),
                    ax=None, title=None, legend=None, legend_col=5, dpi=200, save_location=None, show_figure=True,
                    contour_args={}, number_args={}):
    """
    Plots the ROIs on the input image

    :param rois:
    :param dims:
    :param img_in:
    :param color_list:
    :param start_index:
    :param display_numbers:
    :param thr:
    :param alpha:
    :param axes_off:
    :param figure_size:
    :param ax:
    :param title:
    :param legend:
    :param legend_col:
    :param dpi:
    :param save_location:
    :param show_figure:
    :param contour_args:
    :param number_args:
    """
    n_sessions = len(rois)
    if n_sessions == 1:
        colors = color_list
    else:
        colors = color_list[1:]

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
        output_plot = True
    else:
        output_plot = False

    if img_in is None:
        img_in = np.reshape(np.array(rois[0].mean(1)), dims, order='F')
    img_in[img_in < 0] = 0

    img = label_regions(rois, img_in, dims, alpha, thr, colors=colors)

    # plot the labeled background
    ax.imshow(img)

    # Plot ROI contours for each session
    for ind, A in enumerate(rois):
        # argument validation
        if 'csc_matrix' not in str(type(A)):
            A = csc_matrix(A)

        nR = A.shape[1]  # number of components

        # get the contours for the components
        coordinates = get_contours(A, img_in.shape, thr=thr)

        d1, d2 = dims
        coms = get_centroids(A, d1, d2)  # center of mass for components

        # assign color to ROIs
        color = colors[ind]

        # plot the component contours
        for c in coordinates:
            v = c['coordinates']
            ax.plot(*v.T, color=color, **contour_args)
        if display_numbers:
            for i in range(nR):
                ax.text(coms[i, 1], coms[i, 0], str(i+start_index), color=color, **number_args)

    if axes_off:
        ax.axis('off')

    if legend is not None and n_sessions > 1:
        while len(legend) < n_sessions:
            legend.append('Session ' + str(n_sessions-len(legend)+1))
        for i in range(n_sessions):
            ax.plot([], [], color=color_list[i + 1], label=legend[i], linewidth=2)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 0), ncol=legend_col)

    # Add suptitle and make layout tight
    if title is not None:
        ax.set(title=title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    else:
        plt.tight_layout()

    if save_location is not None:
        plt.savefig(save_location, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)

    if output_plot and show_figure:
        plt.show(block=False)
    elif output_plot is False:
        pass
    else:
        plt.close()


def plot_roi_crossregistration(rois_in, rois_msp, dims, img_in=None, color_list=DEFAULT_COLORS, start_index=0,
                               display_numbers=False, thr=0.9, alpha=0.2, axes_off=True,
                               figure_size=(8, 6), title=None, legend=None, dpi=200, save_location=None,
                               show_figure=True, contour_args={}, number_args={}):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figure_size)
    plot_roi_masked(rois_in, dims, img_in=img_in, color_list=color_list, start_index=start_index,
                    display_numbers=False, thr=thr, alpha=alpha, axes_off=axes_off,
                    ax=axs[0], title='Sessions', legend=legend, legend_col=3, dpi=dpi,
                    save_location=None, show_figure=True,
                    contour_args=contour_args, number_args=number_args)

    if 'list' not in str(type(rois_msp)):
        rois_msp_list = [rois_msp]
    else:
        rois_msp_list = rois_msp

    plot_roi_masked(rois_msp_list, dims, img_in=img_in, color_list=color_list, start_index=start_index,
                    display_numbers=display_numbers, thr=thr, alpha=alpha, axes_off=axes_off,
                    ax=axs[1], title='MSP', legend=None, dpi=dpi, save_location=None, show_figure=True,
                    contour_args=contour_args, number_args=number_args)

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    if save_location is not None:
        plt.savefig(save_location, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_rois(A, dims=(256, 256), img=None, idChannelList=None, nChannels=None,
              colorList=['#3399cc', '#ff3366'], max_number=None, start_index=0,
              cmap=None, vmin=None, vmax=None, title=None, legend=None,
              display_numbers=False, thr=0.9, axes_off=False,
              figureSize=(8, 8), savePath=None, showFig=True, dpi=200,
              contour_args={}, number_args={}):
    """
    Plots the contours of up to 2 channels with different colors on the same plot.

    Parameters
    ----------
    A : np.ndarray or sparse matrix
        Matrix of Spatial components (d x K).

    dims : Tuple of ints, optional
        (width, height) of image. The default is (256,256).

    img : TYPE, optional
        Background image to display. If not provided, background is generated from components.

    idChannelList : List of lists, optional
        List of list of indices identifying elements in the each of the channel.
        If none, then all entries belong to the same channel. The default is None.

    nChannels : INT, optional
        Number of channels to plot. If None, then 1 channel is being used . The default is None.

    colorList : TYPE, optional
        List with the colors for the channels. The default is ['#3399cc','#ff3366'].

    max_number : INT, optional
        maximum number of  components to plot and number. The default is None.

    cmap : Colormap, optional
        Color map used for the background. The default is None.

    vmin : FLOAT, optional
        Minimum value in background plot. The default is None.
    vmax : FLOAT, optional
        Maximum value in background plot. The default is None.

    title : STR, optional
        Title for the plot. The default is False.

    legend : list of STR, optional
        List with legend for each channel. The default is False.

    display_numbers : Boolean, optional
        Flag for displaying component number in the channel. The default is False.

    thr : FLOAT, optional
        Threshold for the energy content within the contour. The default is 0.9.

    axes_off : Boolean, optional
        Flag for displaying axis ticks/units. The default is False.

    contour_args : TYPE, optional
        DESCRIPTION. The default is {}.

    number_args : TYPE, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    None.

    """

    # argument validation
    if 'csc_matrix' not in str(type(A)):
        A = csc_matrix(A)

    # create a default background if none provided
    if img is None:
        img = np.reshape(np.array(A.mean(1)), dims, order='F')

    nR = A.shape[1]  # number of components
    if max_number is None:
        max_number = nR

    # If no channel lists are provided, set all ROIs to same channel
    if idChannelList is None:
        idChannelList = [[i for i in range(nR)]]
    else:
        for i, l in enumerate(idChannelList):
            if not isinstance(l, list):
                idChannelList[i] = l.tolist()

    if nChannels is None:
        nChannels = 1

    # Check whether nChannels corresponds to number of channel lists
    if nChannels > len(idChannelList):
        lRem = []
        for l in idChannelList:
            lRem += l
        idChannelList.append(list(set(range(nR)) - set(lRem)))

    while len(idChannelList) < nChannels:
        idChannelList.append([])

    while len(colorList) < nChannels:
        colorList.append('#abcdef')

    # get the contours for the components
    coordinates = get_contours(A, img.shape, thr=thr)

    d1, d2 = np.shape(img)
    coms = get_centroids(A, d1, d2)  # center of mass for components

    # assign color to each trace
    colorInd = np.ones((nR,), dtype=int) * (nChannels - 1)

    for i, ch in enumerate(idChannelList):
        colorInd[ch] = i

    colors = [colorList[c] for c in colorInd]

    plt.figure(figsize=figureSize)
    plt.ioff()
    ax = plt.gca()
    # plot the background
    if vmax is None or vmin is None:
        plt.imshow(img, interpolation=None, cmap=cmap,
                   vmin=np.percentile(img[~np.isnan(img)], 0.5),
                   vmax=np.percentile(img[~np.isnan(img)], 99.5))
    else:
        plt.imshow(img, interpolation=None, cmap=cmap, vmin=vmin, vmax=vmax)

    # plot the component contours in each channel/group
    for i, (c, col) in enumerate(zip(coordinates, colors)):
        v = c['coordinates']
        if i < max_number:
            plt.plot(*v.T, c=col, **contour_args)
            # conditionally plot component number
            if display_numbers:
                if 'color' in number_args.keys():
                    ax.text(coms[i, 1], coms[i, 0], str(i + start_index), **number_args)
                else:
                    ax.text(coms[i, 1], coms[i, 0], str(i + start_index), color=col, **number_args)
        else:
            break

    if axes_off:
        ax.axis('off')

    if title is not None:
        plt.title(title)

    if legend is not None and nChannels > 1:
        while len(legend) < nChannels:
            legend.append('Data ' + str(nChannels - len(legend) + 1))
        for i in range(nChannels):
            plt.plot([], [], color=colorList[i], label=legend[i], linewidth=2)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

    plt.tight_layout()

    if savePath is not None:
        plt.savefig(savePath, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
    if showFig:
        plt.show(block=False)
    else:
        plt.close()
    plt.ion()

    return


def create_image_montage(images, centroids=None, figureSize=(12, 12), title=None,
                         showIndex=False, palette=None, showFig=True, savePath=None):
    """
    Creates a montage of the images given.

    Parameters
    ----------
    images : List
        List of images for montaging.
    centroids : array [nElemnts x 2], optional
        Array of the center of mass for each neuron. Rows are [Cx, Cy]. The default is None.
    figureSize : tuple (scalar, scalar), optional
        Dimensions of output figure in inches. The default is (12,12).
    title : str, optional
        Sup title used for the montage. The default is None.
    showIndex : Bool, optional
        flag for showing titles for each subplot. If centroids are given the title
        includes '(Cx, Cy)', otherwise it is only the index of the image list. The default is False.
    palette : colormap, optional
        Colormap used for plotting the montage. If none is provided, 'viridis' with nan
        shown as red is used. The default is None.
    showFig : Boolean, optional
        Flag to display the plot on screen. The default is False.
    savePath : str, optional
        Path (with filename) to save the plot. The default is False.

    Returns
    -------
    None.

    """

    # get the number of images and grid size
    nImages = len(images)
    nCols = np.ceil(np.sqrt(nImages)).astype('int')
    nRows = np.ceil(nImages / nCols).astype('int')
    plt.ioff()
    fig, axs = plt.subplots(ncols=nCols, nrows=nRows, figsize=figureSize)
    if palette is None:
        palette = copy(plt.cm.viridis)
        palette.set_bad('r')

    for ind, image in enumerate(images):
        axs[np.unravel_index(ind, (nRows, nCols))].imshow(image, cmap=palette)
        axs[np.unravel_index(ind, (nRows, nCols))].axis('off')
        # add image title
        if showIndex:
            if centroids is not None:
                titleStr = '{:0}: ({:1.0f}, {:0.0f})'.format(ind, centroids[ind, 0], centroids[ind, 1])
            else:
                titleStr = str(ind)
            axs[np.unravel_index(ind, (nRows, nCols))].set(title=titleStr)
    # Switch off the axes for grid locations without images
    for ind in range(nImages, nCols * nRows):
        axs[np.unravel_index(ind, (nRows, nCols))].axis('off')
    # Add suptitle and make layout tight
    if title is not None:
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        plt.tight_layout()

    if savePath is not None:
        plt.savefig(savePath, bbox_inches='tight', pad_inches=0, transparent=True)
    if showFig:
        plt.show(block=False)
    else:
        plt.close()
    plt.ion()


def get_neuron_images(data, dims, win_dims, centroids=None, padding=True):
    """
    Generates a list of images with dimension winDims centered around each of centroids

    Parameters
    ----------
    data : csc_matrix or ndarray [npix x nchannels]
        Spatial footprint matrix.
    dims : tuple (X,Y)
        Dimensions of field of view.
    win_dims : tuple (X,Y)
        Dimensions of window centeed ator each component.
    centroids : List, optional
        List of centroid around which windows are centered. The default is None.
    padding : Bool, optional
        Flag to have all windows same size. Padded area is replaced . The default is True.

    Returns
    -------
    images : list of ndarray
        List of images centered at centroids of spatial footprints.

    """

    if 'csc_matrix' not in str(type(data)):
        data = csc_matrix(data)

    images = []
    d1, d2 = dims
    if centroids is None:
        centroids = get_centroids(data, d1, d2)
        centroids = np.flip(centroids, axis=1)

    halfwin = np.ceil(np.divide(win_dims, 2)).astype('int')
    for neuron, cent in zip(data.T, centroids):
        neuron = neuron.toarray().reshape(dims, order='F')
        cent = np.around(cent).astype('int')

        boundsX = [cent[0] - halfwin[0], cent[0] + halfwin[0]]
        boundsY = [cent[1] - halfwin[1], cent[1] + halfwin[1]]

        if boundsX[0] < 0:
            indX = list(range(0, boundsX[1]))
            indimgX = list(range(win_dims[0] - len(indX), win_dims[0]))
        elif boundsX[1] > d1:
            indX = list(range(boundsX[0], d1))
            indimgX = list(range(0, len(indX)))
        else:
            indX = list(range(boundsX[0], boundsX[1]))
            indimgX = list(range(0, win_dims[0]))

        if boundsY[0] < 0:
            indY = list(range(0, boundsY[1]))
            indimgY = list(range(win_dims[1] - len(indY), win_dims[1]))
        elif boundsY[1] > d2:
            indY = list(range(boundsY[0], d2))
            indimgY = list(range(0, len(indY)))
        else:
            indY = list(range(boundsY[0], boundsY[1]))
            indimgY = list(range(0, win_dims[1]))

        if padding:
            imgtmp = np.empty(win_dims)
            imgtmp.fill(np.nan)
            imgtmp[np.ix_(indimgY, indimgX)] = neuron[np.ix_(indY, indX)]
        else:
            imgtmp = neuron[np.ix_(indY, indX)]
        images.append(imgtmp)

    return images


def get_contours(A, dims, thr=0.9, swap_dim=False):
    """Gets contour of spatial components and returns their coordinates
    Method of thresholding: keeps the pixels that contribute up to a specified fraction of the energy
     Args:
         A:   np.ndarray or sparse of Spatial components (d x K)
         dims: tuple of ints
               Spatial dimensions of movie (x, y)
         thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)

     Returns:
         Coor: list of coordinates with center of mass and
                contour plot coordinates (per layer) for each component
    """

    if 'csc_matrix' not in str(type(A)):
        A = csc_matrix(A)
    d, nr = np.shape(A)
    d1, d2 = dims

    coordinates = []
    # for each patches
    for i in range(nr):
        # we compute the cumulative sum of the energy of the Ath component that has been ordered from least to highest
        patch_data = A.data[A.indptr[i]:A.indptr[i + 1]]
        indx = np.argsort(patch_data)[::-1]

        cumEn = np.cumsum(patch_data[indx]**2)
        if len(cumEn) == 0:
            pars = dict(
                coordinates=np.array([]),
                neuron_id=i,
            )
            coordinates.append(pars)
            continue
        else:
            # we work with normalized values
            cumEn /= cumEn[-1]
            Bvec = np.ones(d)
            # we put it in a similar matrix
            Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]][indx]] = cumEn

        if swap_dim:
            Bmat = np.reshape(Bvec, dims, order='C')
        else:
            Bmat = np.reshape(Bvec, dims, order='F')

        # for each dimension we draw the contour
        vertices = find_contours(Bmat.T, thr)
        # this fix is necessary for having disjoint figures and borders plotted correctly
        v = np.atleast_2d([np.nan, np.nan])
        for _, vtx in enumerate(vertices):
            num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
            if num_close_coords < 2:
                if num_close_coords == 0:
                    # case angle
                    newpt = np.round(old_div(vtx[-1, :], [d2, d1])) * [d2, d1]
                    vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)
                else:
                    # case one is border
                    vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
            v = np.concatenate(
                (v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)
        pars = {'coordinates': v,
                'neuron_id': i }
        coordinates.append(pars)
    return coordinates
