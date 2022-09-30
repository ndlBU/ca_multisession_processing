#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Created on 01-Sep-2022 at 4:42 PM

# @author: jad
"""
import sys
import glob
import logging
import os
import time

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import numpy as np
import matplotlib.pyplot as plt

import msp as msp
from utilities.utilities import dict_to_csc, load_dict_from_hdf5
from utilities.visualization import plot_roi_crossregistration, plot_roi_masked


#%%
def main():
    pass

    #%% SETUP
    # Specify the experiment parameters
    number_of_sessions = 5
    dims = (256, 256)
    registration_offsets = (32, 32)

    # specify the location of the data
    root_path = '/'
    # Specify the save location
    save_path = '/'
    # Specify the experiment name
    exp = 'Experiment_name_'
    # Specify the session names
    session_names = ['S' + '{:02n}'.format(i) for i in range(number_of_sessions)]

    session_files = []
    Ain = []
    for session in session_files:
        # Load estimated session ROIs
        # Append estimated session ROIs to Ain as csc_matrix shape[d x neurons]
        Ain.append(None)

    templates = []
    template_files = []
    for template in template_files:
        # Load estimated session templates

        # Append session templates to templates
        templates.append(None)

    # optional load background image
    img_bck = None

    #%% CROSS REGISTRATION
    # Create MSP object
    m = msp.MSP(dims, registration_offsets, Ain, templates, save_path, exp, session_names=session_names)

    # Update parameters as required
    # params = {
    #     'energy_threshold': 0.9,
    #     'max_centroid_dist': (6, 6),
    #     'spatial_corr_threshold': 0.6,
    #     'edge_dist': (6, 6),
    #     'dilation_kernel': 5,
    #     'erosion_kernel': 5,
    #     'median_filter_size': 5,
    # }
    # m.update(params)

    # Cross register and save sessions
    m.cross_register(save=True)

    # plot aligned ROIs
    plot_roi_masked(m.rois_in, m.dims, img_in=img_bck, thr=0.9, alpha=0.2, axes_off=True, figure_size=(8, 8),
                    title=None, legend=session_names, dpi=200, save_location=None, show_figure=True,
                    contour_args={}, number_args={})

    # plot crossreqistration results
    plot_roi_crossregistration(m.rois_in, [m.rois_msp_aligned[0]], m.dims, img_in=img_bck,
                               start_index=0, display_numbers=False, thr=0.9, alpha=0.25, axes_off=True,
                               title=None, legend=session_names, dpi=200, save_location=None, figure_size=(8, 5),
                               show_figure=True, contour_args={}, number_args={})

    #%% SAVING MSP
    # Save the cross registered MSP
    m.save_msp(os.path.join(save_path,'test.hdf5'))

    #%% LOADING MSP
    # Load and plot saved MSP
    m_new = msp.load_msp(os.path.join(save_path,'test.hdf5'))

    # plot aligned ROIs
    plot_roi_masked(m_new.rois_in, m_new.dims, img_in=img_bck, thr=0.9, alpha=0.2, axes_off=True, figure_size=(8, 8),
                    title=None, legend=session_names, dpi=200, save_location=None, show_figure=True,
                    contour_args={}, number_args={})

    # plot crossreqistration results
    plot_roi_crossregistration(m_new.rois_in, [m_new.rois_msp_aligned[0]], m_new.dims, img_in=img_bck,
                               start_index=0, display_numbers=False, thr=0.9, alpha=0.25, axes_off=True,
                               title=None, legend=session_names, dpi=200, save_location=None, figure_size=(8, 5),
                               show_figure=True, contour_args={}, number_args={})

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
#
