# Arquivo main do projeto.
import argparse
from bdb import BdbQuit
import os
import traceback as tcb
import json

from doc_ock.mp_lock import mp_lock

from utils import paths
import text_2_image

from anonymize_input import load_annotations
from utils.logging_cfg import logging


def process_main(input_fpath):
    img_id = input_fpath.sample_id()
    anon_labels_fpath = input_fpath.labels_fpath()
    json_arq = load_annotations(str(anon_labels_fpath).replace(".json", ".bg.json"))
    try:
        for it_idx in range(n_iters):
            text_2_image.control_mask_gen(input_fpath, json_arq)
    except Exception as e:
        logging.error(' '.join((
            f"Caught exception {e} at synthesis iteration index {it_idx}",
            f"when processing image {img_id} with annotation {anon_labels_fpath}",
            f"\n{tcb.format_exc()}"
        )))
        raise e

def synthesize_dir(cfg, synth_dir):
    global n_iters
    n_iters = cfg["num-iters"]

    try:
        fnames = list(synth_dir.list_input_images())
        if not cfg["exec-parallel"]:
            logging.info("Executing serially")
            [process_main(input_fpath) for input_fpath in fnames]
        else:
            n = min(cfg["num-max-procs"], len(fnames))
            mp_lock(
                fnames, process_main, save_callback=None, num_procs=n,
                out_path=(synth_dir.path_mpout / "synthesize_bgs").as_posix(),
                shared_data={'n_iters': n_iters}
            )
    except KeyboardInterrupt:
        exit(-1)
    except BdbQuit:
        os._exit(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./config.json')
    parser.add_argument(
        "--sample-label", default="front", type=str,
        choices=list(map(str, paths.SampleLabel.__members__.values()))
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"\"{args.config}\" is not a readable file.")

    with open(args.config) as f:
        cfg = json.load(f)

    synth_dir = paths.SynthesisDir(args.sample_label)

    synthesize_dir(cfg, synth_dir)
