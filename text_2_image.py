# Arquivo que insere as informações falsas nas imagens.

import itertools
import json
import os
import random
import secrets
import string
import time
from pathlib import Path

import cv2 as cv
from PIL import Image, ImageDraw, ImageFont

from utils import class_pessoa
# import polygon_functions

from utils.logging_cfg import logging


logging.getLogger("PIL.Image").setLevel(logging.CRITICAL+1)

entities = json.loads(Path('files/entities.json').read_text())


# TODO: estimate these heights from annotation
height_dict = {
    'nome': 4.6,
    'nomePai': 3.6, 'nomeMae': 3.6,
    'date': 3.1, 'city-est': 3.1,
    'serial?': 3.5, 'cod-sec': 2.9
}


# Gera o texto a ser colocado na mask.
def text_generator(tipo_texto, pessoa, control_text):
    qtd_chars = control_text
    text = ''
    if tipo_texto in ('nome', 'filiacao1', 'filiacao2'):
        text = pessoa.set_nome(qtd_chars)
    elif tipo_texto == 'cpf':
        text = pessoa.set_cpf()
    elif tipo_texto == 'rg':
        text = pessoa.set_rg()
    elif tipo_texto == 'orgaoexp':
        text = pessoa.set_org()
    elif tipo_texto in 'naturalidade':
        text = pessoa.set_cid_est(qtd_chars)
    elif tipo_texto == 'rg_org_est':
        text = pessoa.set_rg_org_est()
    elif tipo_texto == 'datanasc':
        text = pessoa.set_datanasc()
    elif tipo_texto == 'dataexp':
        text = pessoa.set_dataexp()
    elif tipo_texto == 'rh':
        text = pessoa.set_fator_rh()
    elif tipo_texto == 'obs':
        text = pessoa.set_obs()
    elif tipo_texto == 'reg_civil':
        text = pessoa.set_folha()
    elif tipo_texto == 'aspa':
        text = pessoa.set_aspa()
    elif tipo_texto == 'via':
        text = pessoa.set_via()
    elif tipo_texto == 'pis':
        text = pessoa.set_pis(qtd_chars)
    elif tipo_texto == 'cod_4':
        text = pessoa.set_cod_4()
    elif tipo_texto == '5-code':
        text = pessoa.set_n_5()
    elif tipo_texto == 'cod_10':
        text = pessoa.set_cod_10()
    elif tipo_texto == 'cid':
        text = pessoa.set_cid(qtd_chars)
    elif tipo_texto == 'cod_8':
        text = pessoa.set_cod_8()
    elif tipo_texto == 'n_via':
        text = pessoa.set_n_via()
    elif tipo_texto == 'n_6':
        text = pessoa.set_n_6()
    elif tipo_texto == 'serial':
        text = f"{''.join(map(str, (random.randint(0, 8) for _ in range(4))))}-{random.randint(0, 8)}"
    elif tipo_texto == 'codsec':
        text = secrets.token_hex(4).upper()
    return text


def med_text_area(text_width, text_height):
    if text_height > text_width:
        return text_width
    a = text_width / (text_height * 0.6)
    return int(a)


def localize_text_area(temp_mask_path):
    """Attempts to shrink annotated ROI by detecting text."""
    x, y, w, h = 0, 0, 0, 0
    temp_mask = cv.imread(temp_mask_path)
    gray = cv.cvtColor(temp_mask, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)

    thresh = cv.dilate(thresh, None, iterations=15)
    thresh = cv.erode(thresh, None, iterations=15)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w >= 5 and h >= 5:
            return [x, y, w, h]
    return [x, y, w, h]


# Gera as demais masks.
def text_mask_generator(json_arq, img_spath):
    synth_dir = img_spath.synth_dir
    img_fname = img_spath.name
    area_n_text = []
    bg_color = 'white'
    draw_anchor = None
    draw_align = 'center'
    p1 = class_pessoa.Person()

    with Image.open(str(synth_dir.path_input / img_fname)) as img:
        img_width, img_height = img.size

    with Image.open(str(synth_dir.path_anon / img_fname)) as img:
        bimg_width, bimg_height = img.size

    mask = Image.new('RGB', (img_width, img_height), color=bg_color)
    mask_name = 'mask_' + img_fname
    mask.save(os.path.join(synth_dir.path_mask, mask_name))
    mask.close()

    temp_mask_path = Path(synth_dir.path_mask) / ('temp_mask_' + img_fname)

    regions = json_arq

    # Checa se a imagem está no path
    if regions is not None:
        qtd_regions = len(regions)
        for aux in range(qtd_regions):
            print(regions[aux])
            mask_open = Image.open(synth_dir.path_mask / mask_name)

            tag = regions[aux]['region_attributes']['tag']
            if regions[aux]['region_attributes']['info_type'] == 'p' and \
                    len(regions[aux]['region_attributes']) > 1:

                tipo_texto = regions[aux]['region_attributes']['text_type']

                font_color = (8, 8, 8)
                if tipo_texto in ('nome', 'serial', 'datanasc'):
                    font_type = (synth_dir.path_static / 'fonts' / 'tahoma' / 'tahoma-bold.ttf').as_posix()
                else:
                    font_type = (synth_dir.path_static / 'fonts' / 'tahoma' / 'tahoma-3.ttf').as_posix()

                if regions[aux]['region_shape_attributes']['name'] == 'rect':
                    # Região é um retângulo.
                    x_inicial = regions[aux]['region_shape_attributes']['x']
                    width = regions[aux]['region_shape_attributes']['width']
                    y_inicial = regions[aux]['region_shape_attributes']['y']
                    height = regions[aux]['region_shape_attributes']['height']

                    x_final = x_inicial + width
                    y_final = y_inicial + height

                    width = x_final - x_inicial
                    height = y_final - y_inicial

                    min_x, max_x, min_y, max_y = tuple(map(int, (x_inicial, x_final, y_inicial, y_final)))

                else:
                    # Não é um retângulo.
                    all_points_x = regions[aux]['region_shape_attributes']['all_points_x']
                    all_points_y = regions[aux]['region_shape_attributes']['all_points_y']
                    min_x, min_y, max_x, max_y = tuple(map(
                        int,
                        (min(all_points_x), min(all_points_y), max(all_points_x), max(all_points_y))))
                    width = max_x - min_x
                    height = max_y - min_y

                if tipo_texto in height_dict:
                    height = int(bimg_height * height_dict[tipo_texto] / 100)

                qtd_chars = med_text_area(width, height)
                font = ImageFont.truetype(font_type, height)
                text = text_generator(tipo_texto, p1, control_text=qtd_chars)
                ImageDraw.Draw(mask_open).text(
                    (min_x, min_y), text, font_color, anchor=draw_anchor, font=font, align=draw_align)

                if tipo_texto != 'x':
                    temp_mask = Image.new('RGB', (img_width, img_height), color=bg_color)
                    ImageDraw.Draw(temp_mask).text(
                        (min_x, min_y), text, font_color, anchor=draw_anchor, font=font, align=draw_align)
                    temp_mask.save(temp_mask_path)
                    temp_mask.close()

                    area = localize_text_area(temp_mask_path.as_posix())
                    area.append(text)
                    area.append(tag)
                    area_n_text.append(area)
                    os.remove(temp_mask_path)

                mask_open.save(os.path.join(synth_dir.path_mask, mask_name))
                mask_open.close()

            else:  # Texto default do documento
                transcription = regions[aux]['region_attributes']['transcription']

                # Região é um retângulo
                if regions[aux]['region_shape_attributes']['name'] == 'rect':
                    x_inicial = regions[aux]['region_shape_attributes']['x']
                    width = regions[aux]['region_shape_attributes']['width']
                    y_inicial = regions[aux]['region_shape_attributes']['y']
                    height = regions[aux]['region_shape_attributes']['height']

                    x_final = x_inicial + width
                    y_final = y_inicial + height

                    min_x, max_x, min_y, max_y = x_inicial, x_final, y_inicial, y_final

                    if transcription != 'X':
                        area = [min_x, min_y, width, height, transcription, tag]
                        area_n_text.append(area)

                else:
                    all_points_x = regions[aux]['region_shape_attributes']['all_points_x']
                    all_points_y = regions[aux]['region_shape_attributes']['all_points_y']

                    width = -1
                    height = -1

                    if transcription != 'X':
                        area = [all_points_x, all_points_y, width, height, transcription, tag]
                        area_n_text.append(area)

    return area_n_text


# Cria o txt baseado nas possíveis rotações que ocorreram com a imagem
def write_txt_file(txt_name, area_n_text, synth_dir):
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
    im.save(str(synth_dir.path_output / f'{txt_name}_mask_GT.jpg'))
    with open(synth_dir.path_output / f'{txt_name}_GT.txt', 'w') as file:
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


# Faz a função de main() desse arquivo.
def control_mask_gen(img_spath, json_arq):
    inicio = time.time()
    area_n_text = text_mask_generator(json_arq, img_spath)
    logging.debug(f"Execution time for mask gen: {str(time.time() - inicio)}")
    mult_img(
        f"mask_{img_spath.name}", img_spath, area_n_text=area_n_text, param=150)
