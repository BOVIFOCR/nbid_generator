import copy
import text_2_image

# from anonymize_input import load_annotations
# from utils.logging_cfg import logging

def process_from_image(inst, mode, n_iters):
    ret = {}
    for it_idx in range(n_iters):
        synthed = text_2_image.ctrl_mask_gen_from_inst(inst)
        new_name = text_2_image.create_img_name(inst['name'])
        ret[new_name] = {"image": synthed, "labels": copy.deepcopy(inst['new_labels'])}
    return ret
