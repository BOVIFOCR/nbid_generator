#pylint: disable=E1101
#pylint: disable=E1102

'''
Main file for RG images annonimization 

Author: Luiz Coelho
Data: April 2023
'''

import os
import glob
import json
import numpy as np
import cv2

from warping import (resize_image_and_annotation,
                     warp_image_and_annotation,
                     rewarp_image)
from masking import mask_fields, mask_regions
from gan_model import load_gan_model
from impainting import inpaint_gan


def anonimize_single(image_path, annotation_path):
    '''
    Anonimize a single image and save in output folder 

    Parameters:
        image_path (str): Path for image
        annotation_path (str): Path for annotation json

    Returns:
        binary_sum (str): Binary string of the sum of a and b
    '''

    # Read files
    # Using cv2.IMREAD_UNCHANGED to ignore EXIF and so that annotation
    # makes sense
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    with open(annotation_path, encoding="utf-8") as file_handler:
        annotation_json = json.load(file_handler)
    print(image_path, annotation_path)

    # Resize
    image, annotation_json = resize_image_and_annotation(image, annotation_json,
                                                          MAX_IMAGE_SIZE)

    # Crop document and warp image
    warped_image, annotation_json, warp_matrix = warp_image_and_annotation(image,
                                                              annotation_json)
    if warped_image is None:
        return None
    cv2.imwrite(os.path.join("warped_input/", os.path.basename(image_path)),
                warped_image)

    # Create masked image
    anonymization_mask = mask_fields(warped_image, annotation_json)
    mask_full = np.zeros_like(anonymization_mask)
    mask_full = cv2.bitwise_or(mask_full, anonymization_mask)
    cv2.imwrite(os.path.join("masked_image/", os.path.basename(image_path)),
                mask_full)
    image_masked = np.bitwise_and(warped_image, cv2.bitwise_not(mask_full)[..., np.newaxis])
    cv2.imwrite(os.path.join("masked_warped/", os.path.basename(image_path)),
                image_masked)

    # Text field inpainting
    gan, mpv = load_gan_model()
    inpainted_warped = inpaint_gan(
                warped_image, mask_full, gan, mpv)
    cv2.imwrite(os.path.join("impainted_warped/", os.path.basename(image_path)),
                inpainted_warped)

    # Other region impainting (face and signature)
    classic_impainting_mask = mask_regions(inpainted_warped, annotation_json,
                                            fields=['face'])
    warped_telea = inpaint_gan(
                inpainted_warped, classic_impainting_mask, gan, mpv)
    classic_impainting_mask = mask_regions(image_masked, annotation_json,
                                            fields=['assinatura'])
    anonymized_warped = inpaint_gan(
                warped_telea, classic_impainting_mask, gan, mpv)
    cv2.imwrite(os.path.join("anonymized_warped/", os.path.basename(image_path)),
                anonymized_warped)

    # Rewarping
    rewarped_full = rewarp_image(image, anonymized_warped, warp_matrix)
    cv2.imwrite(os.path.join("anonymized_rewarped/", os.path.basename(image_path)),
                rewarped_full)


if __name__ == "__main__":

    # Definitions
    IMAGES_FOLDER = "synthesis_input/front/images/"
    ANNOTATIONS_FOLDER = "jsons/"
    MAX_IMAGE_SIZE = 1920

    # List directories
    valid_extensions = ['png', 'jpg', 'jpeg']
    image_file_list = glob.glob(IMAGES_FOLDER+"*")
    image_file_list = [path for path in image_file_list if path.split('.')[-1]
                       in valid_extensions]
    image_file_dict = {os.path.basename(path).split('.')[0]: path
                       for path in image_file_list}
    annotation_file_list = glob.glob(os.path.join(ANNOTATIONS_FOLDER+"*.json"))
    annotation_file_dict = {os.path.basename(path).split('.')[0]: path
                       for path in annotation_file_list}
    # Sync image and annotation files
    image_file_list = sorted([image_file_dict[key] for key
                              in annotation_file_dict.keys()
                              if key in image_file_dict])
    annotation_file_list = sorted([annotation_file_dict[key] for key
                                   in annotation_file_dict.keys()
                                   if key in image_file_dict])

    print(len(image_file_list))
    print(len(annotation_file_list))

    # Run
    for index, (image_file, annotation_file) in enumerate(zip(image_file_list,
                                                annotation_file_list)):
        print("Index: ", index)
        anonimize_single(image_file, annotation_file)
