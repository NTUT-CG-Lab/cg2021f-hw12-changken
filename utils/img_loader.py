from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def list_sorted_dir(dir):
    return sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0]))


def load_img(path):
    return rescale_img(np.array(Image.open(path)))


def rescale_img(image):
    """Rescale image to fit [0, 255] range."""
    if np.max(image) > np.min(image):
        return (np.maximum(image, 0) / image.max()) * 255.0
    return image * 0.


def threshold_img(x, ratio):
    x[x < ratio] = 0
    x[x > ratio] = 1
    return x


def show_pred_mask(image, pred):
    fig, axis = plt.subplots(1, 1, figsize=(10, 10))
    axis.set_title('prediction')
    axis.imshow(image, cmap='gray')
    axis.imshow(pred, cmap='viridis', alpha=0.3)
    plt.show()


def write_pred_mask(image, pred, file_name):
    image = np.array(image)
    pred = np.array(pred)
    image = image.astype(np.uint8)
    pred = pred / 255.
    no_label = (image * (1 - pred)).astype(np.uint8)
    rgb_img = np.stack([image, no_label, no_label], -1)
    rgb_img = Image.fromarray(rgb_img)
    rgb_img.save(file_name)

def write_pred_label_mask(image, pred, label, file_name):
    '''
    output red color is predict mask,
    green color is gt mask,
    yellow color is the intersection of prediction mask and gt mask
    '''
    image = np.array(image)
    pred = np.array(pred)
    label = np.array(label)
    image = image.astype(np.uint8)

    pred = pred / 255.
    label = label / 255.

    pred_mask = (image * pred).astype(np.uint8)
    label_mask = (image * label).astype(np.uint8)

    union_mask = (image * np.logical_or(pred_mask, label_mask)).astype(np.uint8)

    no_mask = image - union_mask

    r = no_mask + pred_mask
    g = no_mask + label_mask
    b = no_mask

    rgb_img = np.stack([r, g, b], -1)
    rgb_img = Image.fromarray(rgb_img)
    rgb_img.save(file_name)