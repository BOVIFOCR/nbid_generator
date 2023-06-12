# Arquivo que insere as informações falsas nas imagens.

import itertools
import json
import os
import random
import secrets
import string
import time
import numpy as np
from pathlib import Path

import cv2 as cv
from PIL import Image, ImageDraw, ImageFont

from utils import class_pessoa
from anonymizer import to_img
from utils.warping import rewarp_image, rotate_all, rewarp_annot
# import polygon_functions

from utils.logging_cfg import logging


logging.getLogger("PIL.Image").setLevel(logging.CRITICAL+1)

entities = json.loads(Path('files/entities.json').read_text())


# TODO: estimate these heights from annotation
height_dict = {
    'nome': 4.6,
    'filiacao1': 3.6, 'filiacao2': 3.6,
    'date': 3.1, 'city-est': 3.1,
    'serial': 3.5, 'cod-sec': 2.9
}

# Empirically estimated
def get_ratio(entity):
    if entity == 'nome':
        return 17/425
    elif entity in ('filiacao1', 'filiacao2', 'rg', 'regcivil'):
        return 15/425
    else:
        return 14/425

def save_annot_txt(labels, save_path):
    regs = labels['regions']

    with open(save_path, 'w') as fd:
        for idx,reg in enumerate(regs):
            pts = reg['points']
            x1, y1, x2, y2 = round(pts[0]), round(pts[1]), round(pts[2]), round(pts[1])
            x3, y3, x4, y4 = round(pts[2]), round(pts[3]), round(pts[0]), round(pts[3])

            fd.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                    idx,
                    x1, y1, x2, y2, x3, y3, x4, y4,
                    reg['transcription'].replace(',', '.'), reg['tag']))

# Gera o texto a ser colocado na mask.
def text_generator(tipo_texto, pessoa):
    if tipo_texto == 'serial':
        text = f"{''.join(map(str, (random.randint(0, 8) for _ in range(4))))}-{random.randint(0, 8)}"
    elif tipo_texto == 'codsec':
        text = secrets.token_hex(4).upper()
    elif tipo_texto in entities and entities[tipo_texto]['is_entity']:
        text = pessoa.get_entity(tipo_texto)
    else:
        text = ""

    if text is None:
        raise ValueError("Entidade não reconhecida: " + tipo_texto)

    return text


def med_text_area(text_width, text_height):
    if text_height > text_width:
        return text_width
    a = text_width / (text_height * 0.6)
    return int(a)

def gen_text_area(region, pessoa, img_height):
    tag = region['tag']

    font_color = (8, 8, 8)

    all_points_x = []
    all_points_y = []
    for point in region['points'][0]:
        all_points_x.append(point[0])
        all_points_y.append(point[1])

    min_x, min_y, max_x, max_y = min(all_points_x), max(all_points_y[0:1]), max(all_points_x), max(all_points_y)
    width = max_x - min_x
    height = max_y - min(all_points_y)

    text = text_generator(tag, pessoa)
    qtd_chars = med_text_area(width, height)
    
    ret_coors = {
        'min_x': min_x, 'min_y': min_y,
        'max_x': max_x, 'max_y': max_y,
        'width': width, 'height': round(get_ratio(region['tag'])*img_height),
        'height_orig': height
    }
    return ret_coors, text


def mask_gen_from_inst(inst):
    pessoa = class_pessoa.Person()
    to_img(inst)
    img_width, img_height = inst['anonymized'].size
    font_type = "./files/fonts/arial/arial-mt-bold.ttf"

    mask = Image.new('RGB', (img_width*3, img_height*3), color='white')
    mask_name = 'mask_' + inst['name']
    area_n_text = []

    for reg in inst['anon_json']['regions']:
        if reg['tag'] in ('assinatura', 'doc', 'face', 'polegar', 'other'):
            continue

        coors, text = gen_text_area(reg, pessoa, img_height)
        if text == "":
            continue

        font = ImageFont.truetype(font_type, coors['height']*3)
        font_regular = ImageFont.truetype(font_type, coors['height'])
        dr = ImageDraw.Draw(mask)
        dr.fontmode = "1"
        dr.text((coors['min_x']*3, coors['min_y']*3),\
                text, (20,27,20), anchor=None, font=font, align='center')

        temp_mask = Image.new('RGB', (img_width, img_height*3), color='black')
        ImageDraw.Draw(temp_mask).text((coors['min_x'], coors['min_y']),\
                text, 'white', anchor=None, font=font_regular, align='center')
    
        area_n_text.append([text, temp_mask.getbbox(), reg['tag']])

    mask = mask.resize((img_width, img_height), Image.ANTIALIAS)
    
    inst['mask'] = mask
    inst['new_labels'] = area_n_text
    return area_n_text

    
def mult_img_from_inst(inst, area_n_text, param):
    back = cv.cvtColor(np.array(inst['anonymized']), cv.COLOR_RGB2BGR)
    new_img_name = create_img_name(inst['name'])
    mask = cv.cvtColor(np.array(inst['mask']), cv.COLOR_RGB2BGR)

    blue_back, green_back, red_back = cv.split(back)
    y = back.shape[0]
    x = back.shape[1]
    blue_mask, green_mask, red_mask = cv.split(mask)
    for j, i in itertools.product(range(y), range(x)):
        if blue_mask[j][i] < param and green_mask[j][i] < param and red_mask[j][i] < param:
            blue_back[j][i] = blue_mask[j][i]
            green_back[j][i] = green_mask[j][i]
            red_back[j][i] = red_mask[j][i]
    # final_img = cv.merge((blue_back, green_back, red_back))
    final_img = cv.merge((red_back, green_back, blue_back))

    inst['synthed'] = final_img


def ctrl_mask_gen_from_inst(inst):
    area_n_text = mask_gen_from_inst(inst)
    mult_img_from_inst(inst, area_n_text=area_n_text, param=150)
    rewarped_full, inst['new_labels'] = rewarp_image(np.array(inst['original']), inst['synthed'],
                            inst['matrix'], inst['new_labels'], inst['name'])

    rewarped_full, inst['new_labels'] = rotate_all(rewarped_full,
                                inst['new_labels'], (360-inst['degrees']) % 360, synthed=True)
    return rewarped_full

# Cria o txt baseado nas possíveis rotações que ocorreram com a imagem
def write_txt_file(txt_name, area_n_text, dir_save):
    txt_text = ''
    with Image.open(str(synth_dir.path_output / (txt_name + '.jpg'))) as img:
        img_width, img_height = img.size
    im = Image.new('RGB', (img_width, img_height), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    for element in area_n_text:
        width = element[2]
        height = element[3]
        tag = element[5]
        if "is_entity" in entities[tag] and not entities[tag]["is_entity"]:
            continue
        transcription = entities[tag].get('transcript', element[4])
        if width == -1 and height == -1:
            x_points = element[0]
            y_points = element[1]

            xy = [(x_points[a], y_points[a]) for a in range(len(x_points))]

            txt_text = txt_text + \
                '{}, {}, {}, {}, {}, {}\n'.format(x_points, y_points, width, height, transcription, tag)

            draw.polygon(xy, fill=(255, 255, 255))
        else:
            x_inicial = element[0]
            y_inicial = element[1]
            x_final = x_inicial + width
            y_final = y_inicial + height

            width = x_final - x_inicial
            height = y_final - y_inicial
            txt_text = txt_text + \
                '{}, {}, {}, {}, {}, {}\n'.format(x_inicial, y_inicial, width, height, transcription, tag)

            draw.rectangle((x_inicial, y_inicial, x_final, y_final), fill=(255, 255, 255))
    im.save(str(dir_save / f'{txt_name}_mask_GT.jpg'))
    with open(dir_save / f'{txt_name}_GT.txt', 'w') as file:
        file.write('x, y, width, height, transcription, tag\n')
        file.write(txt_text)


# Cria um nome aleatório para as imagens geradas.
def create_img_name(img_name):
    num = ''
    random.seed()
    let = ''.join(random.choice(string.ascii_letters) for _ in range(7))
    random.seed()
    for _ in range(7):
        num = num + str(random.randrange(10))
    return img_name + '_' + num + let


# Faz a multiplicação da mask com a imagem original.
def mult_img(mask_name, img_spath, area_n_text, param):
    img_name = img_spath.name
    synth_dir = img_spath.synth_dir
    new_img_name = create_img_name(img_name)
    back = cv.imread(os.path.join(synth_dir.path_anon, img_name))
    blue_back, green_back, red_back = cv.split(back)
    y = back.shape[0]
    x = back.shape[1]
    mask = cv.imread(os.path.join(synth_dir.path_mask, mask_name))
    blue_mask, green_mask, red_mask = cv.split(mask)
    for j, i in itertools.product(range(y), range(x)):
        if blue_mask[j][i] < param and green_mask[j][i] < param and red_mask[j][i] < param:
            blue_back[j][i] = blue_mask[j][i]
            green_back[j][i] = green_mask[j][i]
            red_back[j][i] = red_mask[j][i]
    final_img = cv.merge((blue_back, green_back, red_back))
    outfpath = str(synth_dir.path_output / (new_img_name + '.jpg'))
    cv.imwrite(outfpath, final_img)
    write_txt_file(new_img_name, area_n_text, synth_dir)
    logging.info(f'Synthesis output stored at {outfpath}')
    return new_img_name
