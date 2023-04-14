#pylint: disable=E1101

'''
Auxiliary file for masking functions

Author: Luiz Coelho
Data: April 2023
'''

import cv2
import numpy as np


def mask_regions(image, annotation_json, fields=None):
    '''
    Return anonymization mask for large non-textual regions 
    '''
    if fields is None:
        fields = ['face', 'assinatura']
    anonymization_mask = np.zeros(image.shape[:2]).astype(np.uint8)
    for region in annotation_json[1:]:
        if region['region_shape_attributes']['name'] in fields:
            coords = region['region_shape_attributes']['points']
            anonymization_mask = mask_polygon(anonymization_mask, coords)
    return anonymization_mask

def mask_fields(image, annotation_json):
    '''
    Mask given fields with coordenates
    '''
    fields = ['nome', 'filiacao1', 'filiacao2', 'datanasc', 'naturalidade',
              'orgaoexp', 'codsec', 'serial', 'rh', 'obs']
    anonymization_mask = np.zeros(image.shape[:2]).astype(np.uint8)
    for region in annotation_json[1:]:
        if region['region_shape_attributes']['name'] in fields:
            coords = region['region_shape_attributes']['points']
            anonymization_mask = mask_polygon(anonymization_mask, coords)
    return anonymization_mask

def mask_polygon(masked_image, points):
    '''
    Fill region of polygon in the image, given the coordenates
    '''
    masked_image = cv2.fillPoly(masked_image,
                                   np.array([points], dtype=np.int32), 255)
    return masked_image
        