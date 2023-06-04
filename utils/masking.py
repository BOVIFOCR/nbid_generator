#pylint: disable=E1101

'''
Auxiliary file for masking functions

Author: Luiz Coelho
Data: April 2023
'''

import cv2
import numpy as np


def mask_region(image, annotation_json, image_field_name):
    '''
    Return anonymization mask for large non-textual regions 
    '''
    anonymization_mask = np.zeros(image.shape[:2]).astype(np.uint8)
    for region in annotation_json['regions']:
        if region['tag'] == image_field_name:
            coords = region['points']
            anonymization_mask = mask_polygon(anonymization_mask, coords)
    return anonymization_mask

def mask_fields(image, annotation_json, fields):
    '''
    Mask given fields with coordenates
    '''
    anonymization_mask = np.zeros(image.shape[:2]).astype(np.uint8)
    for region in annotation_json['regions']:
        if region['tag'] in fields:
            coords = region['points']
            anonymization_mask = mask_polygon(anonymization_mask, coords)
    return anonymization_mask

def mask_polygon(masked_image, points):
    '''
    Fill region of polygon in the image, given the coordenates
    '''
    masked_image = cv2.fillPoly(masked_image,
                                   np.array([points], dtype=np.int32), 255)
    return masked_image
