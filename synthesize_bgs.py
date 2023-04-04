# Arquivo main do projeto.
import argparse
from bdb import BdbQuit
import os
import traceback as tcb

from doc_ock.mp_lock import mp_lock

from utils import paths
import text_2_image

from annotation_utils import load_annotations
from logging_cfg import logging

parser = argparse.ArgumentParser()
parser.add_argument('--sample-label', default='front', choices=paths.SampleLabel.__members__.values())
parser.add_argument("--num-iters", default=1)
parser.add_argument("--num-max-procs", default=16)
args = parser.parse_args()


def process_main(input_fpath):
    img_id = input_fpath.sample_id()
    anon_labels_fpath = input_fpath.labels_fpath()
    json_arq = load_annotations(str(anon_labels_fpath).replace(".json", ".bg.json"))
    try:
        for it_idx in range(args.num_iters):
            text_2_image.control_mask_gen(input_fpath, json_arq)
    except Exception as e:
        logging.error(' '.join((
            f"Caught exception {e} at synthesis iteration index {it_idx}",
            f"when processing image {img_id} with annotation {anon_labels_fpath}",
            f"\n{tcb.format_exc()}"
        )))
        raise e


try:
    synth_dir = paths.SynthesisDir(args.sample_label)
    fnames = list(synth_dir.list_input_images())
    if os.environ.get("SINGLE_THREAD"):
        logging.info("Executing serially")
        [process_main(input_fpath) for input_fpath in fnames]
    else:
        n = min(args.num_max_procs, len(fnames))
        mp_lock(
            fnames, process_main, save_callback=None, num_procs=n,
            out_path=(synth_dir.path_mpout / "synthesize_bgs").as_posix()
        )
except KeyboardInterrupt:
    exit(-1)
except BdbQuit:
    os._exit(-1)
