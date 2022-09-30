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

    #%%
    root_path = '/Users/jad/Desktop/caData/processed_supporting/CG045/210306/'
    save_path = '/Users/jad/Desktop/tmp/'
    exp = 'CG045_210306_'
    sessions = [os.path.join(root_path,exp+'S'+'{:02n}'.format((i))+'_initialEstimates.hdf5') for i in range(1,6)]
    fluo_files = [os.path.join(root_path, exp + 'S' + '{:02n}'.format((i)) + '_fluoStats.hdf5') for i in range(1, 6)]

    session_names = ['S' + '{:02n}'.format(i) for i in range(1, 6)]

    Ain = []

    for session in sessions:
        cnm = load_dict_from_hdf5(session)
        Ain.append(dict_to_csc(cnm['estimates']['A']))

    templates = []
    cns = []
    for fluo in fluo_files:
        templates.append(load_dict_from_hdf5(fluo)['template'])
        cns.append(load_dict_from_hdf5(fluo)['Cn'])

    #%%
    m = msp.MSP((256,256), (32,32), Ain, templates, save_path, exp, session_names=session_names)
    m.cross_register()
    #%%
    m.save_msp(os.path.join(save_path,'test.hdf5'))

    #%%
    plot_roi_masked(m.rois_in, m.dims, img_in=cns[0], start_index=0,
                    display_numbers=False, thr=0.9, alpha=0.2, axes_off=True, figure_size=(8, 8),
                    ax=None, title=None, legend=session_names, dpi=200, save_location=None, show_figure=True,
                    contour_args={}, number_args={})

    plot_roi_crossregistration(m.rois_in, [m.rois_msp_aligned[0]], m.dims, img_in=cns[0],
                               start_index=0,
                               display_numbers=False, thr=0.9, alpha=0.25, axes_off=True,
                               title=None, legend=session_names, dpi=200, save_location=None, figure_size=(8, 5),
                               show_figure=True, contour_args={}, number_args={})
    #%%
    m_new = msp.load_msp(os.path.join(save_path,'test.hdf5'))

    plot_roi_masked(m_new.rois_in, m_new.dims, img_in=cns[0], start_index=0,
                    display_numbers=False, thr=0.9, alpha=0.2, axes_off=True, figure_size=(8, 8),
                    ax=None, title=None, legend=session_names, dpi=200, save_location=None, show_figure=True,
                    contour_args={}, number_args={})

    plot_roi_crossregistration(m_new.rois_in, [m_new.rois_msp_aligned[0]], m_new.dims, img_in=cns[0],
                               start_index=0,
                               display_numbers=False, thr=0.9, alpha=0.25, axes_off=True,
                               title=None, legend=session_names, dpi=200, save_location=None, figure_size=(8, 5),
                               show_figure=True, contour_args={}, number_args={})

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
#
