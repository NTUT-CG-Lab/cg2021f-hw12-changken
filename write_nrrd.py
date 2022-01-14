import os
import numpy as np
import nrrd
from PIL import Image
from utils.img_loader import list_sorted_dir


def remove_artifacts(mask):
    mask[mask < 240] = 0  # remove artifacts
    mask[mask >= 240] = 255
    return mask


def write_nrrd(file_name, imgs):
    # normalize imgs
    # TODO

    # set header
    # TODO

    # write nrrd file
    # nrrd.write(file_name, imgs, header)
    # TODO


def write_pred_nrrd(img_dir, file_name):
    '''write out single nrrd file by predict images'''

    # sorted img files
    # TODO

    # read imgs and remove artifacts
    # TODO

    # reverse imgs
    # TODO

    # write out nrrd file
    # TODO


if __name__ == '__main__':
    out_dir = 'data/nrrd_data'
    os.makedirs(os.path.join(out_dir), exist_ok=True)

    target = 'ID00423637202312137826377'

    # write gt mask nrrd
    gt_mask_dir = f'data/pred_data/gt_masks/{target}'
    write_pred_nrrd(gt_mask_dir, f'{out_dir}/{target}_gt_mask.nrrd')

    # write pred mask nrrd
    pred_mask_dir = f'data/pred_data/masks/{target}'
    write_pred_nrrd(pred_mask_dir, f'{out_dir}/{target}_mask.nrrd')