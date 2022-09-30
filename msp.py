#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:50:47 2022

@author: jad
"""

from builtins import object
from builtins import str

import numpy as np
import scipy.stats as stats
from scipy.sparse import csc_matrix, save_npz
import os

import logging
from utilities.utilities import save_dict_to_hdf5, load_dict_from_hdf5, dict_to_csc
from utilities.fluo_processing import cleanup_rois, get_spatial_mask, get_centroids


class MSP(object):

    def __init__(self, dims: (int, int), registration_offset: (int, int),
                 rois_in, alignment_templates, save_location, experiment_name, session_names=None,
                 energy_threshold=0.9, max_centroid_dist=4, spatial_corr_threshold=0.60, edge_dist=(6, 6),
                 dilation_kernel=5, erosion_kernel=5, median_filter_size=5,
                 ):
        """
        Creates an MSP object and initializes it

        :param dims: fov dimension in pixels (int, int)
        :param registration_offset: search offsets for alignment in pixels (int, int)
        :param rois_in: list of spatial footprints each in csc_matrix format
        :param alignment_templates: list of images for aligning sessions
        :param save_location: folder used for saving results (str)
        :param experiment_name: prefix used in naming files (str)
        :param session_names: list of session names (list of str)
        :param energy_threshold: scalar between 0 and 1 corresponding to energy threshold
         for computing contours (default 0.9). Keeps the pixels that contribute up to thr of the energy
        :param max_centroid_dist: maximum distance (pixels) between roi centroids corresponding to same
        roi in different session
        :param spatial_corr_threshold: minimum correlation for classification as same neuron
        :param edge_dist: minimum distance (pixels) to edge of fov in all sessions
        :param dilation_kernel: used for smoothing roi (int)
        :param erosion_kernel: used for smoothing roi (int)
        :param median_filter_size: used for smoothing roi (int)
        """
        # required inputs
        self.dims = dims
        self.registration_offset = registration_offset
        self.rois_in = rois_in
        self.alignment_templates = alignment_templates
        self.save_location = save_location
        self.experiment_name = experiment_name

        # optional session names
        self.session_names = session_names

        # filtering and registration parameters
        self.energy_threshold = energy_threshold
        self.max_centroid_dist = max_centroid_dist
        self.spatial_corr_threshold = spatial_corr_threshold
        self.edge_dist = edge_dist
        self.dilation_kernel = dilation_kernel
        self.erosion_kernel = erosion_kernel
        self.median_filter_size = median_filter_size

        # alignments that will be calculated or set explicitly
        self.alignment_matrix = None
        self.alignment_offsets = None

        # outputs
        self.rois_msp = None
        self.rois_msp_aligned = []
        self.neuron_activity = None
        self.neuron_centroids = None
        self.neuron_counts = None
        self.neuron_registration = {}
        self.roi_files = None
        self.neuron_registration_file = None

        # dimensions of image space used for registration
        self._max_offset = max(registration_offset)
        self._registration_dims = tuple([int(a + 2 * b) for (a, b) in zip(self.dims, self.registration_offset)])
        self._num_sessions = len(rois_in)

    def get_session_offset_detailed(self, fov1, fov2, image_format=True):
        """
        Finds the offset between 2 sessions, and teh
        Parameters
        ----------
        fov1 : First image, used as reference image
        fov2 : Second image, aligned to the first
        image_format : bool, optional
            Indicates that FOVs are indexed as images (rows,cols) ie (y,x). The default is True.

        Returns
        -------
        offset: tuple (int, int)
            Represents the translation (img2 -> img1) applied to img2 to align its content
            with that of img1. This translation minimizes the means square error of the difference
            between the normalized images. Offset is in the coordinate space of img1
            If rowMajor: offset = (dY,dX)
            If columnMajor: offset = (dX,dY)
        offsetList: 2D ndarray
            Image of mean difference between fov1 and fov2 based on offset.
        """

        if fov1.shape != fov2.shape:
            raise IndexError("Images must have the same dimensions.")

        img1 = stats.zscore(fov1, axis=None)
        img2 = stats.zscore(fov2, axis=None)

        dim0, dim1 = img1.shape

        max_offset = self._max_offset

        offset_list = []

        for dY in range(-max_offset, max_offset + 1):
            y11 = 0 if dY < 0 else dY
            y12 = dim0 + dY if dY < 0 else dim0
            y21 = -dY if dY < 0 else 0
            y22 = dim0 if dY < 0 else dim0 - dY

            for dX in range(-max_offset, max_offset + 1):
                pass
                x11 = 0 if dX < 0 else dX
                x12 = dim1 + dX if dX < 0 else dim1
                x21 = -dX if dX < 0 else 0
                x22 = dim1 if dX < 0 else dim1 - dX

                tmp1 = img1[y11:y12, x11:x12]
                tmp2 = img2[y21:y22, x21:x22]

                offset_list.append(np.mean((tmp1 - tmp2) ** 2, axis=None))

        offset_shape = (2*max_offset+1, 2*max_offset+1)

        offset = np.subtract(np.unravel_index(np.argmin(offset_list), offset_shape),
                             (max_offset, max_offset))

        if not image_format:
            offset = offset[::-1]

        return tuple(offset), np.array(offset_list).reshape(offset_shape, order='c')

    def get_session_offset(self, fov1, fov2, image_format=True):
        """
        Find the alignment between sessions
        Parameters
        ----------
        fov1 : First image, used as reference image
        fov2 : Second image, aligned to the first
        image_format : bool, optional
            Indicates that FOVs are indexed as images (rows,cols) ie (y,x). The default is True.

        Returns
        -------
        offset: tuple (int, int)
        Represents the translation (img2 -> img1) applied to img2 to align its content
        with that of img1. This translation minimizes the means square error of the difference
        between the normalized images. Offset is in the coordinate space of img1
        If rowMajor: offset = (dY,dX)
        If columnMajor: offset = (dX,dY)
        """
        offset, _ = self.get_session_offset_detailed(fov1, fov2, image_format=image_format)

        return offset

    def align_to_registration_space(self, Ain, d0: int, d1: int, shift):
        """
        Translates the spatial footprints from a session's field of view (FOV) reference frame
        to the registration's FOV reference frame.

        Parameters
        ----------
        Ain : csc_matrix [d x nElements], (d = d1*d2)
            column sparse matrix of spatial footprints. Each column is a component
        d0 : int
            height of session FOV in pixels.
        d1 : int
            width of session FOV in pixels.
        shift : (int, int)
            (dy, dx) to translation that shifts the coordinates to the registration's coordinates

        Returns
        -------
        TYPE
            Aout: csc_matrix [registrationX*registrationY x nElements].
            column sparse matrix of spatial footprints in the registration reference frame.
            Each column is a component.
        """
        Aout = np.array([]).reshape(np.prod(self._registration_dims), 0)
        indX = int((self._registration_dims[1] - d1) / 2) + shift[1]
        indY = int((self._registration_dims[0] - d0) / 2) + shift[0]
        B = np.zeros(self._registration_dims)
        for comp in Ain.T:
            B[indY:indY + d0, indX:indX + d1] = comp.toarray().reshape((d0, d1), order='F')
            Aout = np.column_stack((Aout, B.flatten(order='F')))

        Aout = csc_matrix(Aout)
        return Aout

    def align_to_session_space(self, Ain, d0: int, d1: int, shift):
        """
        Translates the spatial footprints from the registration's field of view (FOV) reference frame
        to the session's FOV reference frame.

        Parameters
        ----------
        Ain : csc_matrix [d x nElements], (d = d1*d2)
            column sparse matrix of spatial footprints. Each column is a component
        d0 : int
            height of session FOV in pixels.
        d1 : int
            width of session FOV in pixels.
        shift : (int, int)
            (dy, dx) to translation that shifts the coordinates to the session's reference frame

        Returns
        -------
        Aout : csc_matrix [d x nElements], (d = d1*d2)
            column sparse matrix of spatial footprints. Each column is a component

        """
        Aout = np.array([]).reshape(d0 * d1, 0)
        indX = int((self._registration_dims[1] - d1) / 2) + shift[1]
        indY = int((self._registration_dims[0] - d0) / 2) + shift[0]
        B = np.empty((d0, d1))
        for comp in Ain.T:
            B = comp.toarray().reshape(self._registration_dims, order='F')[indY:indY + d0, indX:indX + d1]
            Aout = np.column_stack((Aout, B.flatten(order='F')))

        Aout = csc_matrix(Aout)
        return Aout

    def test_boundary(self, point, dist):
        """
        Verifies that point [x,y] does not fall within dist[dx, dy] of the boundary
         after alignment in all sessions

        Parameters
        ----------
        point : array [2]
            coordinates [x, y].
        dist : TYPE
            DESCRIPTION.

        Returns
        -------
        within_boundary : bool
            returns True if centroid is located at least dist away from all session boundaries.

        """
        within_boundary = True
        for offset in self.alignment_offsets:
            x = point[0] - offset[0] - (self._registration_dims[0] - self.dims[0]) / 2
            y = point[1] - offset[1] - (self._registration_dims[1] - self.dims[1]) / 2

            if (x < dist[0]) or (abs(x - self.dims[0]) < dist[0])\
                    or (y < dist[1]) or (abs(y - self.dims[1]) < dist[1]):
                within_boundary = False
                break

        return within_boundary

    def cross_register(self, save=True):
        """
        Main function - calls other functions to cross register the set of sessions
        :param save: flag to specify whether to save results
        :return:
            neuron_counts: number of identified neurons
            roi_files: absolute location of saved files that contain the cross registered rois aligned to each session
            neuron_registration_file: location of file containing summary of cross registration
        """
        self.verify_inputs()
        self.find_session_alignments()
        self.register()
        if save:
            self.save_data()

        return self.neuron_counts, self.roi_files, self.neuron_registration_file

    def save_data(self):
        """
        Saves the data:
        npz files containing all the unique ROIs aligned to each session
        hdf5 containing a summary of the cross registration
            'neuronActivity':
            'comRegistration': centroid coordinates [x,y]
            'alignmentVector': offsets required to align all sessions to the reference session (first one)
            'alignmentMatrix': pairwise alignment between all sessions
            'neuronCount': number of unique neurons identified from all sessions
        """
        roi_files = []
        if not os.path.isdir(self.save_location):
            os.makedirs(self.save_location, exist_ok=True)

        # Align the spatial matrix to each session and save
        for session, A_aligned in zip(self.session_names, self.rois_msp_aligned):
            save_file = os.path.join(self.save_location, session + '_seedROI.npz')
            save_npz(save_file, A_aligned)
            roi_files.append(save_file)
        self.roi_files = roi_files

        neuron_registration = {'neuronActivity': self.neuron_activity,
                               'comRegistration': np.flip(np.array(self.neuron_centroids), axis=1),
                               'alignmentVector': self.alignment_offsets,
                               'alignmentMatrix': self.alignment_matrix,
                               'neuronCount': self.neuron_counts}

        self.neuron_registration_file = os.path.join(self.save_location, self.experiment_name + '_neuronRegistration.hdf5')
        save_dict_to_hdf5(neuron_registration, self.neuron_registration_file)

    def register(self):
        """
        Performs the registration of neurons from the aligned sessions
        """
        # Iterate over initial estimates and find unique components. Find centroid_2 of component
        neuron_masks = []
        neuron_centroids = []
        cnt = 0  # count of new components

        neuron_activity = np.empty((0, self._num_sessions), dtype=bool)
        new_activity = np.eye(self._num_sessions).astype('int')

        for sessionInd, (session, A, offset) in enumerate(zip(self.session_names, self.rois_in, self.alignment_offsets)):

            rois_temp = self.align_to_registration_space(A, *self.dims, offset)
            masks_temp = get_spatial_mask(rois_temp, self._registration_dims, thr=self.energy_threshold)
            masks_temp, keep_roi, bad_rois = cleanup_rois(masks_temp, dims=self._registration_dims,
                                                          dilation_kernel=self.dilation_kernel,
                                                          erosion_kernel=self.erosion_kernel,
                                                          median_filter_size=self.median_filter_size)
            masks_temp = masks_temp.toarray() * rois_temp.toarray()
            centroids_temp = get_centroids(masks_temp, *self._registration_dims).squeeze()

            # +ve: index of neuron in list, -1: fragmented ROI, -2: in border zone
            session_activity = np.empty((len(masks_temp.T)))
            session_new_activity = new_activity[sessionInd]

            for index, (mask_check, centroid_check) in enumerate(zip(masks_temp.T, centroids_temp)):
                found = False
                if keep_roi[index] is False:
                    session_activity[index] = -1
                    continue
                for indMask, (mask, centroid_found) in enumerate(zip(neuron_masks, neuron_centroids)):
                    dist = np.linalg.norm(centroid_check - centroid_found)
                    rPearson, pVal = stats.pearsonr(mask_check, mask)

                    if dist < self.max_centroid_dist and rPearson > self.spatial_corr_threshold:
                        found = True
                        session_activity[index] = indMask
                        neuron_activity[index, sessionInd] = 1
                        break
                if found:
                    continue
                elif not self.test_boundary(centroid_check, self.edge_dist):
                    session_activity[index] = -2
                else:
                    neuron_masks.append(mask_check)
                    neuron_centroids.append(centroid_check)
                    session_activity[index] = cnt
                    neuron_activity = np.vstack((neuron_activity, session_new_activity))
                    cnt += 1

            self.neuron_registration[session] = session_activity.astype(int)
            logging.info('Registered: ' + session)

        self.rois_msp = csc_matrix(np.array(neuron_masks).T)
        self.neuron_activity = neuron_activity
        self.neuron_centroids = np.array(neuron_centroids)
        self.neuron_counts = cnt

        # Align the spatial matrix to each session
        for session, offset in zip(self.session_names, self.alignment_offsets):
            A_aligned = self.align_to_session_space(self.rois_msp, *self.dims, offset)
            self.rois_msp_aligned.append(A_aligned)
            logging.info('Aligned: ' + session[:-1])

    def find_session_alignments(self):
        """
        Generates the pairwise alignment between sessions, and the alignment used in registration.
        Registration assumes the first session as the reference.
        """
        # Find the alignment matrix between sessions
        if self.alignment_matrix is None:
            alignment_list = []
            for img1 in self.alignment_templates:
                for img2 in self.alignment_templates:
                    offset = self.get_session_offset(img1, img2)
                    alignment_list.append(tuple(offset))

            self.alignment_matrix = np.array(alignment_list).reshape((self._num_sessions, self._num_sessions, 2))
            logging.info('Generated alignment matrix')

        # Find the alignment vector to reference session(first) if not provided
        if self.alignment_offsets is None:
            self.alignment_offsets = self.alignment_matrix[0]

    def verify_inputs(self):
        """
            Checks that all input lists have the same length, and raises error if not.
            Creates session names if they are not provided.
        """
        #
        len_names = [len(self.session_names)] if self.session_names is not None else []
        len_offset = [len(self.alignment_offsets)] if self.alignment_offsets is not None else []
        len_offset_pairs = list(self.alignment_matrix.shape[:2]) if self.alignment_matrix is not None else []
        list_len = len_names + [len(self.alignment_templates), len(self.rois_in)] + len_offset + len_offset_pairs
        if not all(i == self._num_sessions for i in list_len):
            print("Input list length", list_len)
            raise IndexError("The cross registration input lists should all have the same length.")

        if self.session_names is None:
            self.session_names = ['Session'+str(i) for i in range(self._num_sessions)]

    def save_msp(self, filename):
        """
            Saves MSP object to hdf5 file
            Parameters
            ----------
            filename : location of save file if absolute path
                        otherwise saved at save_location
            Returns
            -------
            None
        """
        if '.hdf5' not in filename:
            raise Exception("File extension not supported saving")

        if os.path.isdir(os.path.dirname(filename)):
            save_dict_to_hdf5(self.__dict__, filename)
        else:
            save_dict_to_hdf5(self.__dict__, os.path.join(self.save_location, os.path.basename(filename)))

    def update(self, params_dict):
        """
            Update the attribute values

            Parameters
            ----------
            params_dict : dictionary of attributes/values

            Returns
            -------
            None
        """
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError('No attribute in MSP named {0}'.format(key))

        self.verify_inputs()


def load_msp(filename):
    """
        Loads saved MSP object from hdf5 file
        Parameters
        ----------
        filename : absolution location of file

        Returns
        -------
        MSP object
    """
    if '.hdf5' in filename:
        msp_dict = load_dict_from_hdf5(filename)

        new_obj = MSP(tuple(msp_dict['dims']),
                      tuple(msp_dict['registration_offset']),
                      [dict_to_csc(v) for v in msp_dict['rois_in'].values()],
                      [v for v in msp_dict['alignment_templates'].values()],
                      msp_dict['save_location'].decode(),
                      msp_dict['experiment_name'].decode()
                      )

        #
        updates = {
            'session_names': [v.decode() for v in msp_dict['session_names'].values()],
            'energy_threshold': msp_dict['energy_threshold'],
            'max_centroid_dist': msp_dict['max_centroid_dist'],
            'spatial_corr_threshold': msp_dict['spatial_corr_threshold'],
            'edge_dist': msp_dict['edge_dist'],
            'dilation_kernel': msp_dict['dilation_kernel'],
            'erosion_kernel': msp_dict['erosion_kernel'],
            'median_filter_size': msp_dict['median_filter_size'],
            'alignment_offsets': msp_dict['alignment_offsets'],
            'alignment_matrix': msp_dict['alignment_matrix'],
            'rois_msp': dict_to_csc(msp_dict['rois_msp']),
            'rois_msp_aligned': [dict_to_csc(v) for v in msp_dict['rois_msp_aligned'].values()],
            'neuron_activity': msp_dict['neuron_activity'],
            'neuron_centroids': msp_dict['neuron_centroids'],
            'neuron_counts': msp_dict['neuron_counts'],
            'neuron_registration': msp_dict['neuron_registration'],
            'roi_files': [v.decode() for v in msp_dict['roi_files'].values()],
            'neuron_registration_file': msp_dict['neuron_registration_file'].decode
        }
        new_obj.update(updates)

        return new_obj
    else:
        raise Exception("File extension not supported for loading")