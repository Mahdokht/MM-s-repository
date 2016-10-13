import os
import struct

import cv2
import numpy as np
from docopt import docopt

import cine
from linLUT import linLUT

import glob, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
from skimage.filters import threshold_otsu
from CineRead import frame_reader
from CineRead import create_raw_array
from CineRead import unpack_10bit
from CineRead import read_header


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))


def inds_not_on_edge_mat(cine_file, header):
    image_edge = []
    count = header['cinefileheader'].ImageCount
    for i in range(count):
        mat_image = frame_reader(cine_file,header, start_frame=i, count = 1)
        imarrayBWnum = np.sum(mat_image, axis=0)
        image_edge.append([imarrayBWnum[0],imarrayBWnum[-1]])
    return np.where(is_outlier(np.array(image_edge))==False)


def inds_not_empty_mat(cine_file, header, obj_thresh):
    width, height = header['bitmapinfoheader'].biWidth, header['bitmapinfoheader'].biHeight
    count = header['cinefileheader'].ImageCount
    temp = np.zeros((height,width), dtype=bool)     
    for i in range(count):
        mat_image = frame_reader(cine_file,header, start_frame=i, count = 1)
        thresh = threshold_otsu(mat_image)
        imbinary = mat_image > thresh
        temp = np.bitwise_or(imbinary,temp)
    imarrayBG = np.sum(temp, axis=0)
    BG_minmax = [min(imarrayBG),max(imarrayBG)]
    image_minmax=[]
    for i in range(count):
        mat_image = frame_reader(cine_file,header, start_frame=i, count = 1)
        thresh = threshold_otsu(mat_image)
        imbinary = mat_image > thresh
        imbinaryBWnum = np.sum(imbinary, axis=0)
        image_minmax.append(np.subtract([min(imbinaryBWnum),max(imbinaryBWnum)],BG_minmax))
    return np.where(np.abs(np.sum(image_minmax,axis=1))>obj_thresh)
      

def select_frame(cine_file, object_threshold):    

    header = read_header(cine_file)
    
    if not object_threshold:    
        object_threshold = 10
        
    notedge = inds_not_on_edge_mat(cine_file, header)
    notempty = inds_not_empty_mat(cine_file, header,object_threshold)
    #return notedge
    not_edge_empty = list(set(np.array(notedge).ravel()).intersection(np.array(notempty).ravel()))
    return not_edge_empty[len(not_edge_empty)/2-1]

