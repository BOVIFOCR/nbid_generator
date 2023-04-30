'''
Main file for RG images anonymization 

Author: Luiz Coelho
Data: April 2023
'''

from anonymizer import Anonymizer

if __name__ == '__main__':

    MAX_IMAGE_SIZE = 1920
    IMAGES_FOLDER = "./synthesis_input/front/images/"# "/home/luizcoelho/datasets/ufpr_documents/front/images/"
    ANNOTATIONS_FOLDER = "./synthesis_input/front/labels/" #"/home/luizcoelho/datasets/ufpr_documents/front/jsons/"
    anon_front = Anonymizer(IMAGES_FOLDER, ANNOTATIONS_FOLDER, 'front', MAX_IMAGE_SIZE)
    anon_front.run()

    # IMAGES_FOLDER = "/home/luizcoelho/datasets/ufpr_documents/back/images/"
    # ANNOTATIONS_FOLDER = "/home/luizcoelho/datasets/ufpr_documents/back/jsons/"
    # anon_back = Anonymizer(IMAGES_FOLDER, ANNOTATIONS_FOLDER, 'back', MAX_IMAGE_SIZE)
    # anon_back.run()
