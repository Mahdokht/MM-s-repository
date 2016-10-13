import os
import struct

import cv2
import numpy as np
from docopt import docopt

import cine
from linLUT import linLUT


def read_header(myfile):
    with open(myfile, 'rb') as f:
        header = {}
        header['cinefileheader'] = cine.CINEFILEHEADER()
        header['bitmapinfoheader'] = cine.BITMAPINFOHEADER()
        header['setup'] = cine.SETUP()
        f.readinto(header['cinefileheader'])
        f.readinto(header['bitmapinfoheader'])
        f.readinto(header['setup'])

        # header_length = ctypes.sizeof(header['cinefileheader'])
        # bitmapinfo_length = ctypes.sizeof(header['bitmapinfoheader'])

        f.seek(header['cinefileheader'].OffImageOffsets)
        header['pImage'] = struct.unpack('{}q'.format(header['cinefileheader'].ImageCount),
                                         f.read(header['cinefileheader'].ImageCount * 8))

    return header


def unpack_10bit(data, width, height):
    packed = np.frombuffer(data, dtype='uint8').astype('uint16')
    unpacked = np.zeros([height, width], dtype='uint16')

    unpacked.flat[::4] = (packed[::5] << 2) | (packed[1::5] >> 6)
    unpacked.flat[1::4] = ((packed[1::5] & 0b00111111) << 4) | (packed[2::5] >> 4)
    unpacked.flat[2::4] = ((packed[2::5] & 0b00001111) << 6) | (packed[3::5] >> 2)
    unpacked.flat[3::4] = ((packed[3::5] & 0b00000011) << 8) | packed[4::5]

    return unpacked


def create_raw_array(data, header):
    width, height = header['bitmapinfoheader'].biWidth, header['bitmapinfoheader'].biHeight
    BayerPatterns = {3: 'gbrg', 4: 'rggb'}
    #pattern = BayerPatterns[header['setup'].CFA]

    #if header['bitmapinfoheader'].biCompression:
    raw_image = unpack_10bit(data, width, height)
        #fix_bad_pixels(raw_image, header['setup'].WhiteLevel, pattern)
    raw_image = linLUT[raw_image].astype(np.uint16)
    raw_image = np.interp(raw_image, [64, 4064], [0, 2**12-1]).astype(np.uint16)
    #else:
    #    raw_image = np.frombuffer(data, dtype='uint16')
    #    raw_image.shape = (height, width)
    #    #fix_bad_pixels(raw_image, header['setup'].WhiteLevel, pattern)
    #    raw_image = np.flipud(raw_image)
    #    raw_image = np.interp(raw_image, [header['setup'].BlackLevel, header['setup'].WhiteLevel],
    #                                     [0, 2**header['setup'].RealBPP-1]).astype(np.uint16)

    return raw_image, width, height



def frame_reader(myfile, header, start_frame=1, count=1):
    frame = start_frame
    with open(myfile, 'rb') as f:
       
        frame_index = frame - 1
        #print "Reading frame {}".format(frame)

        f.seek(header['pImage'][frame_index])

        AnnotationSize = struct.unpack('I', f.read(4))[0]
        Annotation = struct.unpack('{}B'.format(AnnotationSize - 8),
                                           f.read((AnnotationSize - 8) / 8))
        header["Annotation"] = Annotation

        ImageSize = struct.unpack('I', f.read(4))[0]

        data = f.read(ImageSize)

        raw_image, width, height = create_raw_array(data, header)
            
        frame += 1
        count -= 1
    return raw_image.reshape(height,width)

