#pylint: disable=E1101

'''
Auxiliary file for impainting functions

Author: Luiz Coelho
Data: April 2023
'''

import numpy as np
import cv2
import torch
from torchvision import transforms

def blur_roi(image, annotation_json, ksize=(3, 3)):
    '''
    Blur regions of interest
    '''
    fields = ['face', 'assinatura']
    height, width = image.shape[:2]
    for region in annotation_json[1:]:
        if region['region_shape_attributes']['name'] in fields:
            coords = region['region_shape_attributes']['points']
            x_tl= int(min(coords[0][:,0]))
            y_tl = int(min(coords[0][:,1]))
            x_br = int(max(coords[0][:,0]))
            y_br = int(max(coords[0][:,1]))
            x_tl = max(0, x_tl)
            y_tl = max(0, y_tl)
            x_br = min(x_br, width)
            y_br = min(y_br, height)
            image[y_tl:y_br, x_tl:x_br] = cv2.blur(image[y_tl:y_br, x_tl:x_br],
                                          ksize=ksize,
                                          borderType=cv2.BORDER_REPLICATE)
    return image

def inpaint_telea(img, inpaint_mask, inpaint_radius=3):
    """
    Applies Telea's inpainting method (TODO: link OpenCV doc.)
    """
    return cv2.inpaint(
        img, inpaint_mask,
        flags=cv2.INPAINT_TELEA, inpaintRadius=inpaint_radius)

def inpaint_gan(img, mask, model, mpv):
    """
    Run GAN network for inpainting
    """
    img = torch.unsqueeze(transforms.ToTensor()(img), 0).cpu()
    mask = torch.unsqueeze(transforms.ToTensor()(mask), 0).cpu()
    mask.clamp_(0, 1)
    img.sub_(img * mask)
    img.add_(mpv * mask)

    with torch.no_grad():
        output = model(torch.cat((img, mask), dim=1))
    inpainted = poisson_blend(img, output, mask)

    inpainted = inpainted.squeeze()  # not an inplace op
    inpainted.mul_(255).add_(0.5).clamp_(0, 255)

    return inpainted.permute(1, 2, 0).to("cpu", torch.uint8).numpy()

def poisson_blend(input_image, output, mask, infer_center=True):
    """
    * inputs:
        - input_image (torch.Tensor, required)
                Input tensor of Completion Network, whose shape = (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor of Completion Network, whose shape = (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network, whose shape = (N, 1, H, W).
    * returns:
                Output image tensor of shape (N, 3, H, W) inpainted with poisson 
                image editing method.
    """
    input_image = input_image.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask, mask, mask), dim=1)  # convert to 3-channel format
    num_samples = input_image.shape[0]
    ret = []
    for i in range(num_samples):
        # applies `torch -> numpy` conversion to arguments
        dstimg = transforms.functional.to_pil_image(input_image[i])
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]

        srcimg = transforms.functional.to_pil_image(output[i])
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]

        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]

        # compute mask's center
        if infer_center:
            x_list, y_list = [], []
            for j in range(msk.shape[0]):
                for k in range(msk.shape[1]):
                    if msk[j, k, 0] > 0:
                        y_list.append(j)
                        x_list.append(k)
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            center = (xmax + xmin) // 2, (ymax + ymin) // 2
        else:
            center = msk.shape[1] // 2, msk.shape[0] // 2
        dstimg = cv2.inpaint(dstimg, msk[:, :, 0], 1, cv2.INPAINT_TELEA)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret
