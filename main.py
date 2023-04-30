from utils import paths
# from synthesize_bgs import inpaint_dir
# from anonymize_input import anonymize_dir
from anonymizer import Anonymizer

import argparse
import json
import os
import cv2

def main(cfg):
    samples = ['front', 'back']

    for sample in samples:
        input_dir = paths.SynthesisDir(sample)
        print(input_dir.path_input, input_dir.path_json)
        anon = Anonymizer(str(input_dir.path_input) + '/', str(input_dir.path_json) + '/', \
                        mode=sample, max_img_size=cfg['max-width'])

        ret = anon.run(return_anon=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./config.json')
    args = parser.parse_args()
    
    if not os.path.isfile(args.config):
        print(f"\"{args.config}\" is not a readable file.")

    with open(args.config) as f:
        cfg = json.load(f)

    main(cfg)