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
import re
import copy
import numpy as np
import cv2
from PIL import Image

from utils.warping import (resize_image_and_annotation,
                           warp_image_and_annotation,
                           rewarp_image,
                           rotate_all)
from utils.masking import mask_fields, mask_region
from utils.gan_model import load_gan_model
from utils.inpainting import inpaint_gan

def make_inst(name, image, inpainted_warped, warp_matrix, warped_annotation_json, annotation_json, degrees):
    return {
        "name": name,
        "original": image,
        "anonymized": inpainted_warped,
        "matrix": warp_matrix,
        "anon_json": warped_annotation_json,
        "json": annotation_json,
        "degrees": degrees
    }

def to_img(inst):
    if (type(inst['original']) != Image.Image):
        inst['original'] = Image.fromarray(inst['original'])
        inst['anonymized'] = Image.fromarray(inst['anonymized'])

def to_array(inst):
    if (type(inst['original']) != np.ndarray):
        inst['original'] = np.array(inst['original'])
        inst['anonymized'] = np.array(inst['anonymized'])


class Anonymizer():
    '''
    Anonymizer class to handle front and back RG anonimizations
    '''
    def __init__(self, images_folder, annotations_folder, mode, gan_cfg_file,
                filelist=None, inpaint=True, max_img_size=1920):
        if mode not in ['front', 'back']:
            print("Mode must be 'front' or 'back': {mode}")

        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.mode = mode
        self.inpaint = inpaint
        self.max_img_size = max_img_size

        # Load GAN
        if inpaint:
            self.gan, self.mpv = load_gan_model(gan_cfg_file)

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


        if filelist is not None:
            # Get filenames
            self.image_file_list, self.annotation_file_list,\
                self.image_degrees = self.get_from_filelist(filelist)

        else:
            # Get image and annotation paths
            self.image_file_list, self.annotation_file_list = self.list_directories()
            self.image_file_list, self.annotation_file_list = self.sync_directories(
                                    self.image_file_list, self.annotation_file_list)
            self.image_degrees = [0]*len(self.image_file_list)

    def get_from_filelist(self, filelist):        
        with open(filelist, "r") as fd:
            lines = [x.strip('\n') for x in fd.readlines()]

        image_file_list = []
        annotation_file_list = []
        image_degrees = []
        for line in lines:
            splt = line.split(' ')
            image_file_list.append(self.images_folder + splt[0])
            image_degrees.append(int(splt[1]))
            annotation_file_list.append(self.annotations_folder + os.path.splitext(splt[0])[0] + ".json")
        return image_file_list, annotation_file_list, image_degrees

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
        for region in annotation_json['regions']:
            contain_names.append(region['tag'])
        for region in self.image_regions:
            if region not in contain_names:
                return False
        return True

    def anonymize_json(self, annotation_json):
        '''
        Remove information from transcription, keeping the text structure
        Replace all letters for 'a' and replace all numbers for '0'
        '''
        letters_placeholder = 'a'
        numbers_placeholder = '0'
        for idx, region in enumerate(annotation_json['regions']):
            if region['transcription'] is None:
                continue
            transcription = region['transcription']
            if isinstance(transcription, str):
                anon_transcription = re.sub(r'(\w(?<!\d))', letters_placeholder, transcription)
                anon_transcription = re.sub(r'\d', numbers_placeholder, anon_transcription)
                annotation_json['regions'][idx]['transcription'] = anon_transcription
        return annotation_json

    def save_annotation_json(self, annotation_json, annotation_path, save_path):
        '''
        Save json anonymized and warped
        '''
        anon_annotation_json = self.anonymize_json(annotation_json.copy())
        for idx, region in enumerate(anon_annotation_json['regions']):
            if 'points' not in region:
                continue
            points = region['points'].astype(int)
            points = points.tolist()
            anon_annotation_json['regions'][idx]['points'] = points
        with open(os.path.join(save_path, os.path.basename(annotation_path)),
                'w', encoding="utf-8") as file_handler:
            json.dump(annotation_json , file_handler)

    def run(self, return_anon=False):
        """
        Run anonimization on all files
        """
        ret_all = []
        for index, (image_file, annotation_file, degrees) in enumerate(zip(self.image_file_list,
                                                self.annotation_file_list, self.image_degrees)):
            if index <= -1:
                continue
            print(image_file, annotation_file)
            print("Index: ", index)
            ret = self.anonymize_single(image_file, annotation_file, degrees, inpaint=self.inpaint,
                                        return_anon=return_anon)
            # return
            if return_anon:
                yield ret
            else:
                ret_all.append(ret)
        return ret_all

    def anonymize_single(self, image_path, annotation_path, degrees, inpaint=True, return_anon=False):
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

        # Validate json
        valid = self.validate_json(annotation_json)
        if not valid:
            return

        # Rotate
        image, annotation_json = rotate_all(image, annotation_json, degrees)

        # Resize
        image, annotation_json = resize_image_and_annotation(image, annotation_json,
                                                        self.max_img_size)

        # Crop document and warp image
        warped_annotation_json = copy.deepcopy(annotation_json)
        warped_image, warped_annotation_json, warp_matrix = warp_image_and_annotation(image,
                                                                warped_annotation_json)
        if warped_image is None:
            return None

        # Create masked image
        anonymization_mask = mask_fields(warped_image, warped_annotation_json, self.fields)
        mask_full = np.zeros_like(anonymization_mask)
        mask_full = cv2.bitwise_or(mask_full, anonymization_mask)

        if inpaint:
            # Text field inpainting
            inpainted_warped = inpaint_gan(
                        warped_image, mask_full, self.gan, self.mpv)

            # Other region impainting (face, signature, thumb)
            for image_region in self.image_regions:
                image_masked_region = mask_region(inpainted_warped, warped_annotation_json,
                                                        image_region)
                inpainted_warped = inpaint_gan(
                            inpainted_warped, image_masked_region, self.gan, self.mpv)
        else:
            inpainted_warped = warped_image

        if return_anon:
            return make_inst(image_path.split('/')[-1], image,\
                inpainted_warped, warp_matrix, warped_annotation_json, annotation_json, degrees)

        else:
            # Rewarping
            rewarped_full = rewarp_image(image, inpainted_warped, warp_matrix)
            cv2.imwrite(os.path.join(self.rewarped_dir, os.path.basename(image_path)),
                        rewarped_full)

            # Save anonymized jsons
            self.save_annotation_json(warped_annotation_json, annotation_path, self.warped_dir)
            self.save_annotation_json(annotation_json.copy(), annotation_path, self.rewarped_dir)
