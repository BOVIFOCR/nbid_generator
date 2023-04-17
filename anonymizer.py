#pylint: disable=E1101
#pylint: disable=E1102

'''
Anonimizer class definition

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
from masking import mask_fields, mask_region
from gan_model import load_gan_model
from impainting import inpaint_gan

class Anonymizer():
    '''
    Anonymizer class to handle front and back RG anonimizations
    '''
    def __init__(self, images_folder, annotations_folder, mode, max_img_size=1920):
        if mode not in ['front', 'back']:
            print("Mode must be 'front' or 'back': {mode}")

        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.mode = mode
        self.max_img_size = max_img_size

        # Load GAN
        self.gan, self.mpv = load_gan_model()

        # Fields to anonimyze
        self.fields = ['nome', 'filiacao1', 'filiacao2', 'datanasc', 'naturalidade',
              'orgaoexp', 'codsec', 'serial', 'rh', 'obs', 
              'cpf', 'rg', 'dataexp', 'regcivil', 'te', 'cnh', 'cns',
              'ctps', 'serie', 'uf', 'other', 'pis', 'profissional', 'militar']

        # Fields to impaint separetly
        self.image_regions = ['assinatura', 'face'] if mode == 'front' else \
                                                    ['assinatura', 'polegar']

        # Output folders
        self.warped_dir = f"{self.mode}_anonymized_warped"
        self.rewarped_dir = f"{self.mode}_anonymized_rewarped"
        if not os.path.exists(self.warped_dir):
            os.mkdir(self.warped_dir)
        if not os.path.exists(self.rewarped_dir):
            os.mkdir(self.rewarped_dir)

        # Get image and annotation paths
        self.image_file_list, self.annotation_file_list = self.list_directories()
        self.image_file_list, self.annotation_file_list = self.sync_directories(
            self.image_file_list, self.annotation_file_list)

    def list_directories(self):
        '''
        Get list of image files and json annotation files
        '''
        valid_extensions = ['png', 'jpg', 'jpeg']
        image_file_list = glob.glob(self.images_folder+"*")
        image_file_list = [path for path in image_file_list if path.split('.')[-1]
                        in valid_extensions]

        annotation_file_list = glob.glob(os.path.join(self.annotations_folder+"*.json"))

        return image_file_list, annotation_file_list

    def sync_directories(self, image_file_list, annotation_file_list):
        '''
        Sync image file list and annotation file list
        '''
        image_file_dict = {os.path.basename(path).split('.')[0]: path
                        for path in image_file_list}
        annotation_file_dict = {os.path.basename(path).split('.')[0]: path
                       for path in annotation_file_list}
        image_file_list = sorted([image_file_dict[key] for key
                              in annotation_file_dict.keys()
                              if key in image_file_dict])
        annotation_file_list = sorted([annotation_file_dict[key] for key
                                   in annotation_file_dict.keys()
                                   if key in image_file_dict])
        return image_file_list, annotation_file_list

    def validate_json(self, annotation_json):
        '''
        Verify if json has the mandatory fields
        '''
        contain_names = []
        for region in annotation_json[1:]:
            contain_names.append(region['region_shape_attributes']['name'])
        for region in self.image_regions:
            if region not in contain_names:
                return False
        return True

    def run(self):
        """
        Run anonimization on all files
        """
        for index, (image_file, annotation_file) in enumerate(zip(self.image_file_list,
                                                self.annotation_file_list)):
            if index <= -1:
                continue
            print("Index: ", index)
            self.anonymize_single(image_file, annotation_file)

    def anonymize_single(self, image_path, annotation_path):
        """
        Run anonimization in a pair of image and annotation
        """
        # Read files
        # Using cv2.IMREAD_UNCHANGED to ignore EXIF and so that annotation
        # makes sense
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if len(image.shape) > 2 and image.shape[2] == 4:
            #convert the image from RGBA2RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        with open(annotation_path, encoding="utf-8") as file_handler:
            annotation_json = json.load(file_handler)
        print(image_path, annotation_path)

        # Validate json
        valid = self.validate_json(annotation_json)
        if not valid:
            return

        # Resize
        image, annotation_json = resize_image_and_annotation(image, annotation_json,
                                                        self.max_img_size)

        # Crop document and warp image
        warped_image, annotation_json, warp_matrix = warp_image_and_annotation(image,
                                                                annotation_json)
        if warped_image is None:
            return None
        #cv2.imwrite(os.path.join("warped_input/", os.path.basename(image_path)),
        #            warped_image)

        # Create masked image
        anonymization_mask = mask_fields(warped_image, annotation_json, self.fields)
        mask_full = np.zeros_like(anonymization_mask)
        mask_full = cv2.bitwise_or(mask_full, anonymization_mask)
        #cv2.imwrite(os.path.join("masked_image/", os.path.basename(image_path)),
        #            mask_full)
        #image_masked = np.bitwise_and(warped_image, cv2.bitwise_not(mask_full)[..., np.newaxis])
        #cv2.imwrite(os.path.join("masked_warped/", os.path.basename(image_path)),
        #            image_masked)

        # Text field inpainting
        inpainted_warped = inpaint_gan(
                    warped_image, mask_full, self.gan, self.mpv)
        cv2.imwrite(os.path.join("impainted_warped/", os.path.basename(image_path)),
                    inpainted_warped)

        # Other region impainting (face, signature, thumb)
        for image_region in self.image_regions:
            image_masked_region = mask_region(inpainted_warped, annotation_json,
                                                    image_region)
            inpainted_warped = inpaint_gan(
                        inpainted_warped, image_masked_region, self.gan, self.mpv)
        cv2.imwrite(os.path.join(self.warped_dir, os.path.basename(image_path)),
                    inpainted_warped)

        # Rewarping
        rewarped_full = rewarp_image(image, inpainted_warped, warp_matrix)
        cv2.imwrite(os.path.join(self.rewarped_dir, os.path.basename(image_path)),
                    rewarped_full)
