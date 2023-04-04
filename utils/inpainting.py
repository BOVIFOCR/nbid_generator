import cv2
import torch
from torchvision import transforms

from gan_model.utils import poisson_blend

from defs.geometry import Rectifier


def blur_roi(canvas, roi_poly, ksize=(3, 3)):
    rectifier = Rectifier(canvas, roi_poly)
    roi_warped = rectifier.rectify()
    canvas = cv2.blur(canvas, ksize=ksize, borderType=cv2.BORDER_REPLICATE)
    return rectifier.rewarp(roi_warped)


def inpaint_telea(img, inpaint_mask, inpaint_radius=3):
    """Applies Telea's inpainting method (TODO: link OpenCV doc.)
    """
    return cv2.inpaint(
        img, inpaint_mask,
        flags=cv2.INPAINT_TELEA, inpaintRadius=inpaint_radius)


def inpaint_gan(img, mask, model, mpv, device):
    img = torch.unsqueeze(transforms.ToTensor()(img), 0).to(device)
    mask = torch.unsqueeze(transforms.ToTensor()(mask), 0).to(device)
    mask.clamp_(0, 1)
    img.sub_(img * mask)
    img.add_(mpv * mask)

    with torch.no_grad():
        output = model(torch.cat((img, mask), dim=1))
    inpainted = poisson_blend(img, output, mask)

    inpainted = inpainted.squeeze()  # not an inplace op
    inpainted.mul_(255).add_(0.5).clamp_(0, 255)

    return inpainted.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
