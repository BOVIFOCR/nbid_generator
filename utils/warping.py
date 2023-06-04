#pylint: disable=E1101

'''
Auxiliary file for warping functions 
References:
    https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    https://stackoverflow.com/questions/38285229/calculating-aspect-ratio-of-perspective-transform-destination-image


Author: Luiz Coelho
Data: April 2023
'''

import numpy as np
import cv2
from pytesseract import image_to_osd


def resize_image_and_annotation(image, annotation_json: list, max_size: int):
    '''
    Resize image and bounding box annotation, if image is too large
    '''
    width = int(image.shape[1])
    height = int(image.shape[0])
    max_dim = max(width, height)
    if max_dim > max_size:
        scale = max_size / max_dim
        image = cv2.resize(image, (0,0), fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    else:
        scale = 1

    annotation_json['width'] *= scale
    annotation_json['height'] *= scale
    for idx, region in enumerate(annotation_json['regions']):
        if 'points' not in region:
            continue
        # coord_string = str(region['points'])
        coord_numpy = np.array([[float(value) for value in point]
                                    for point in region['points']])
        coord_numpy *= scale
        annotation_json['regions'][idx]['points'] = coord_numpy

    return image, annotation_json

def order_points(pts):
    '''
    Return a list of coordinates that will be ordered
    such that the first entry in the list is the top-left,
    the second entry is the top-right, the third is the
    bottom-right, and the fourth is the bottom-left
    '''
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    axis_sum = pts.sum(axis = 1)
    rect[2] = pts[np.argmin(axis_sum)]
    rect[0] = pts[np.argmax(axis_sum)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def warp_image(image, rect):
    """
    Return rectified image
    """
    (bottom_right, bottom_left, top_left, top_right) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + \
                      ((bottom_right[1] - bottom_left[1]) ** 2))
    width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + \
                      ((top_right[1] - top_left[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + \
                       ((top_right[1] - bottom_right[1]) ** 2))
    height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + \
                       ((top_left[1] - bottom_left[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [max_height - 1, max_width - 1],
        [max_height - 1, 0],
        [0, 0],
        [0, max_width - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_height, max_width))
    return warped, transform_matrix

def rt270anti(bbox, wd):
    nbbox = []
    for x, y in bbox:
        nbbox.append([y, wd - x])
    return np.array(nbbox)

def rt180(bbox, wd, hg):
    nbbox = []
    for x, y in bbox:
        nbbox.append([hg - x, wd - y])
    return np.array(nbbox)

def rt90anti(bbox, hg):
    nbbox = []
    for x, y in bbox:
        nbbox.append([hg - y, x])
    return np.array(nbbox)


def rotate_bbox(bbox, height, width, degrees):
    if degrees == 270:
        return rt90anti(bbox, width)
    elif degrees == 180:
        return rt180(bbox, width, height)
    elif degrees == 90:
        return rt270anti(bbox, height)
    else:
        return bbox

def rotate_img(img, degrees):
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return img

def rotate_all(image, annotation_json, degrees):
    wd, hg = annotation_json['width'], annotation_json['height']

    image = rotate_img(image, degrees)
    for i in range(len(annotation_json['regions'])):
        reg = annotation_json['regions'][i]
        reg['points'] = rotate_bbox(reg['points'], wd, hg, degrees)
 
    if degrees in (90, 270):
        annotation_json['width'] = hg
        annotation_json['height'] = wd

    return image, annotation_json


def warp_image_and_annotation(image, annotation_json: list):
    '''
    Crop and warp document image and annotation coordinates
    '''
    for region in annotation_json['regions']:
        if region["tag"] == "doc":
            doc_coords = region["points"]
            break
    else:
        return None, None, None
    doc_coords = order_points(doc_coords)

    # Warp image
    image_warped, transform_matrix = warp_image(image, doc_coords)
    # Warp annotations
    annotation_json['width'] = image_warped.shape[1]
    annotation_json['height'] = image_warped.shape[0]
    for index, region in enumerate(annotation_json['regions']):
        if 'points' not in region:
            continue
        coord_numpy = annotation_json['regions'][index]['points']
        coord_numpy = cv2.perspectiveTransform(np.array([coord_numpy]),
                                               transform_matrix)
        annotation_json['regions'][index]['points'] = coord_numpy
    return image_warped, annotation_json, transform_matrix

def rewarp_image(original, warped, tranformation_matrix):
    '''
    Reverse the warping operation and stitch rewarped back in original image
    '''
    rewarped = cv2.warpPerspective(warped, tranformation_matrix,
                                   original.shape[1::-1],
                                   flags=cv2.WARP_INVERSE_MAP)
    rewarped_full = np.where(rewarped != (0,0,0), rewarped, original)
    return rewarped_full

def rewarp_annot(inst):
    new_labels = inst['new_labels']
    mat = inst['matrix']
    mat = np.linalg.inv(mat)

    label = []
    for i in range(len(new_labels)):
        bbox = new_labels[i][1]
        x1,y1,x2,y2 = bbox[0], bbox[1], bbox[2], bbox[1]
        x3,y3,x4,y4 = bbox[2], bbox[3], bbox[0], bbox[3]

        bbox = []
        for coord in [[x1, y1, 1], [x2, y2, 1], [x3, y3, 1], [x4, y4, 1]]:
            bbox.append(list(mat @ coord)[:-1])
        label.append({'region_id': i, 'transcription': new_labels[i][0], 
                      'tag': new_labels[i][2], 'points': bbox})

    new_labels = {'regions': label, 'height': inst['original'].size[1],
                                    'width': inst['original'].size[0],
                                    'filename': inst['name']}
    return new_labels

 