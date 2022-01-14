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
    imgs = (imgs.T / 255.).astype(np.float32)

    # set header
    header = {
        'space directions': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        'space origin': [0, 0, 0]
    }

    # write nrrd file
    nrrd.write(file_name, imgs, header)

def write_pred_nrrd(img_dir, file_name):
    '''write out single nrrd file by predict images'''

    # sorted img files
    img_fs = list_sorted_dir(img_dir)

    # read imgs and remove artifacts
    imgs = np.array([remove_artifacts(np.array(Image.open(os.path.join(img_dir, img_f)))) for img_f in img_fs])

    # reverse imgs
    imgs = np.flip(imgs, 0)

    # write out nrrd file
    write_nrrd(file_name, imgs)


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