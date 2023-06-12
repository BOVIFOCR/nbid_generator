from synthesize_bgs import process_from_image
from anonymizer import Anonymizer
from text_2_image import save_annot_txt

import argparse
import json
import os
import cv2

def save_instances(images, final_dir, sample_cfg):
    for im in images:
        cv2.imwrite(final_dir + sample_cfg['output_images'] + im + ".jpg", images[im]['image'])
        save_annot_txt(images[im]['labels'], final_dir + sample_cfg['output_labels'] + im + ".tsv")


def main(cfg):
    samples = cfg['samples']

    base_dir = cfg['input_dir']
    save_dir = cfg['save_dir']

    for sample in samples:
        sample_cfg = cfg['sample_cfg'][sample]
        print(sample_cfg)

        anon = Anonymizer(
                base_dir + sample_cfg['images'],
                base_dir + sample_cfg['labels'],
                sample,
                cfg['gan_config'],
                filelist = base_dir + sample_cfg['filelist'],
                inpaint = cfg['inpaint'],
                max_img_size = cfg['max_width']
            )

        for ret in anon.run(return_anon=True):
            ims = process_from_image(ret, sample, cfg['num_iters'])
            save_instances(ims, save_dir, sample_cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./config.json')
    args = parser.parse_args()
    
    if not os.path.isfile(args.config):
        print(f"\"{args.config}\" is not a readable file.")

    with open(args.config) as f:
        cfg = json.load(f)

    main(cfg)