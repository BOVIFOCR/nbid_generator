from utils import paths
from synthesize_bgs import inpaint_dir
from anonymize_input import anonymize_dir

import argparse
import json
import os

def main(cfg):
    # samples = ['front', 'back']

    # input_dir = paths.SynthesisDir('front')
    # anonymize_dir(cfg, input_dir)

    # inpaint_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./config.json')
    args = parser.parse_args()
    
    if not os.path.isfile(args.config):
        print(f"\"{args.config}\" is not a readable file.")

    with open(args.config) as f:
        cfg = json.load(f)

    main(cfg)