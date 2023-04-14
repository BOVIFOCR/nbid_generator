# -*- coding: latin-1 -*-

import argparse
import json
import math
import os
import sys
import time
import traceback as tcb
import typing
from bdb import BdbQuit
from itertools import islice

import cv2
import doxapy as doxa
import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np

from utils.logging_cfg import logging

from utils import inpainting, paths
from gan_model.models import CompletionNetwork

from defs.geometry import Polygon2D, Rectifier



def resize_image(img, max_width=None):
    width = int(img.shape[1])
    height = int(img.shape[0])

    max_dim = max(width, height)
    if max_dim > max_width:
        scale = max_width / max_dim
        new_dim = tuple(map(int, (width * scale, height * scale)))
        new_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        new_img = img

    return new_img, scale


def load_annotations(labels_fpath, rescale=None, rectifier=None):
    with open(labels_fpath, encoding="utf-8") as json_file:
        json_arq = json.load(json_file)
    if rescale or rectifier:
        rescale = rescale if rescale is not None else 1.

        for idx in range(len(json_arq)):
            pts = [
                [x, y]
                for x, y in zip(
                    json_arq[idx]["region_shape_attributes"]["all_points_x"],
                    json_arq[idx]["region_shape_attributes"]["all_points_y"],
                )
            ]
            num_pts = len(pts)
            pts = np.array([pts], dtype=np.float64)
            if rectifier:
                pts = cv2.perspectiveTransform(pts, rectifier.H)
            pts = rescale * pts
            pts = pts[0]
            json_arq[idx]["region_shape_attributes"]["all_points_x"] = [pt[0] for pt in pts]
            json_arq[idx]["region_shape_attributes"]["all_points_y"] = [pt[1] for pt in pts]
    return json_arq

def load_gan_model(gan_cfg_dir):
    with open(gan_cfg_dir, "r") as f:
        gan_cfg = json.load(f)

    gan_dir = "/".join(gan_cfg_dir.split('/')[:-1]) + "/" + gan_cfg["init_model_cn"]

    gan = CompletionNetwork()
    gan.load_state_dict(torch.load(gan_dir, map_location=gan_cfg["device"]))

    mpv = torch.tensor(gan_cfg["mpv"]).view(3, 1, 1).to(gan_cfg["device"])

    return gan, mpv


def chunk(sequence, chunksize):
    sequence = iter(sequence)
    sub_chunk = list(islice(sequence, chunksize))
    while sub_chunk:
        yield sub_chunk
        sub_chunk = list(islice(sequence, chunksize))


def process_single(img_spath, max_width, gan, mpv):
    try:
        logging.info(f"Processing {img_spath}")

        if img_spath.name.endswith('.png'):
            src_img = cv2.imread(str(img_spath), cv2.IMREAD_COLOR)
        else:
            src_img = cv2.imread(str(img_spath), cv2.IMREAD_UNCHANGED)

        # TODO: implement .add .mul and .sub on Polygon
        # TODO: implement .expand_from_center() on Polygon
        doc_poly: Polygon2D = Polygon2D.from_str(
            sample_df[
                sample_df['filename'] == img_spath.name
            ]['doc_polygon'].iloc[0])
        rectifier: Rectifier = Rectifier(src_img, doc_poly)
        assert rectifier.H is not None

        img = rectifier.rectify()
        img, scale = resize_image(img, max_width=max_width)

        # Loads and resizes annotations, storing it as .bg.json
        labels_fpath = img_spath.labels_fpath()
        # json_arq = load_annotations(labels_fpath, rescale=scale)
        json_arq = load_annotations(
            labels_fpath, rescale=scale, rectifier=rectifier)
        with open(
            str(labels_fpath).replace(".json", ".bg.json"), mode="w", encoding="utf-8"
        ) as json_fp:
            json.dump(json_arq, json_fp)

        # Anonimize input and stores it at `synth_dir.path_anon`
        logging.debug(f"Inpainting {img_spath}")
        inpainted = np.copy(img)

        binarization = doxa.Binarization(doxa.Binarization.Algorithms.SAUVOLA)
        img_bin = np.empty(img.shape[:2], dtype=np.uint8)
        binarization.initialize(cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY))
        binarization.to_binary(img_bin)

        for annot in json_arq:
            entity_label = annot['region_attributes']['tag']
            entity_poly = Polygon2D(list(zip(
                annot['region_shape_attributes']['all_points_x'],
                annot['region_shape_attributes']['all_points_y']))
            )

            inpaint_mask = entity_poly.mask(inpainted.shape)

            # if entity_label in ('polegar', 'face'):
            if 'extend_corner' in entities[entity_label]:
                corner = entities[entity_label]
                bb = entity_poly.bounding_box()
                if 'left' in corner:
                    bb.width += bb.top_left.x
                    bb.top_left.x = 0
                if 'right' in corner:
                    bb.width = inpainted.shape[1] - bb.width
                if 'bottom' in corner:
                    bb.height = inpainted.shape[0] - bb.height
                entity_poly = bb.to_polygon()

                inpaint_mask = entity_poly.mask(inpaint_mask.shape)
                inpainted = inpainting.blur_roi(inpainted, entity_poly)

                inpainted = inpainting.inpaint_telea(
                    inpainted, inpaint_mask, inpaint_radius=1)
            else:
                inpaint_mask = cv2.bitwise_and(
                    inpaint_mask, cv2.bitwise_not(img_bin))

                # blur + dilate iterations
                # (otherwise, character borders are not covered)
                blur_shape = (3, 3)
                dilation_shape = cv2.MORPH_ELLIPSE
                dilatation_size = 2
                dilation_structure = cv2.getStructuringElement(
                    dilation_shape,
                    (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                    (dilatation_size, dilatation_size))
                blur_dilate_iters = 1
                for _ in range(blur_dilate_iters):
                    inpaint_mask = cv2.blur(inpaint_mask, blur_shape)
                    inpaint_mask = cv2.dilate(inpaint_mask, dilation_structure)

            inpainted = inpainting.inpaint_gan(
                inpainted, inpaint_mask, gan, mpv, device)
            inpainted = inpainting.blur_roi(
                inpainted, entity_poly, ksize=(5, 5))

        warped_outpath = img_spath.warped_fpath().as_posix()
        cv2.imwrite(warped_outpath, inpainted)
        logging.info(f"Anonimized warped ROI stored at {warped_outpath}")

        rewarped = rectifier.rewarp(cv2.resize(
            inpainted, rectifier.rect_shape))
        anon_outpath = img_spath.anon_fpath().as_posix()
        cv2.imwrite(anon_outpath, rewarped)
        logging.info(f"Anonimization output stored at {anon_outpath}")

    except Exception as e:
        logging.error("\n".join((
            f"Caught exception {e} when processing {img_spath}:",
            f"{tcb.format_exc()}"
        )))
        raise e


def process_list(cfg, sdir, input_list, gan, mpv):
    global sample_df, entities
    sample_df = pd.read_csv(
        sdir.path_input_base / 'sample.via.csv',
        usecols=('filename', 'doc_polygon'))
    entities = json.loads((
        sdir.path_input_base / 'entities.json').read_text())

    for img_spath in input_list:
        process_single(img_spath, cfg["max-width"], gan, mpv)


def wait_for_job_completion(jobs: typing.List[mp.Process], wait_period=0.25):
    while len(jobs) > 0:
        for idx, job in enumerate(jobs):
            if not job.is_alive():
                return idx
        time.sleep(wait_period)


def process_parallel(cfg, synth_dir, gan, mpv):
    # loads input data
    input_list = list(synth_dir.list_input_images(
        for_anon=True, randomize=True))[:cfg["num-max-samples"]]
    logging.info((
        f"Finished loading {len(input_list)} ",
        "images for anonimization"))

    # sets multiprocessing via torch
    num_input_items = len(input_list)
    num_procs = min(cfg["num-max-procs"], num_input_items)
    job_chunksize = min(
        cfg["job-chunksize"], math.ceil(num_input_items / num_procs))

    mp.set_start_method("spawn")
    torch.set_num_threads(1)

    gan.share_memory()
    mpv.share_memory_()

    # launches jobs
    logging.info(
        " ".join((
                f"Distributing execution along {num_procs} threads",
                f"processing {job_chunksize} images each",)))
    jobs: typing.List[mp.Process] = []
    try:
        for input_chunk in chunk(input_list, job_chunksize):
            # launches 1 job per given number of processes
            if len(jobs) >= num_procs:
                logging.debug("Waiting for a job to complete")
                finished_job_id = wait_for_job_completion(jobs)
                jobs[finished_job_id].close()
            job = mp.Process(target=process_list, args=[cfg, synth_dir, input_chunk, gan, mpv])
            job.start()
            jobs.append(job)
        logging.info(f"Joining all {len(jobs)} jobs")
        for job in jobs:
            job.join()
            job.close()
            del job
    except KeyboardInterrupt:
        sys.stderr('\n'.join((
            "Caught KeyboardInterrupt",
            "To exit, stop with <CTRL+Z>, and terminate with `kill %1`")))


def anonymize_dir(cfg, synth_dir):
    gan, mpv = load_gan_model(cfg["gan-config"])

    if not cfg["exec-parallel"]:
        logging.info("Executing serially")
        
        # loads input data
        input_list = list(synth_dir.list_input_images(
            for_anon=True, randomize=True))[:cfg["num-max-samples"]]
        logging.info((
            f"Finished loading {len(input_list)} ",
            "images for anonimization"))

        process_list(input_list, cfg, synth_dir, gan, mpv)
    else:
        process_parallel(cfg, synth_dir, gan, mpv)


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

    try:
        anonymize_dir(cfg)
    except BdbQuit:
        logging.error(f"Leaving for {BdbQuit}")
        os._exit(-2)
    except Exception as e:
        logging.error(f"Caught {type(e)} {e}\nExiting")
        logging.error(tcb.print_exc())
        exit(-3)
