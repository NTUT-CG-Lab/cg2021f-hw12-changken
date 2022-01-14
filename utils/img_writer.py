import matplotlib.pyplot as plt
import os
from utils.img_loader import list_sorted_dir, load_img


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def save_imgs(img_dir, file_name, cols=10):
    fs = list_sorted_dir(img_dir)
    imgs = [load_img(os.path.join(img_dir, f)) for f in fs]

    n = len(imgs)
    rows = (n // cols) + 1
    figure, axs = plt.subplots(rows, cols, figsize=(20, 20))
    axs = trim_axs(axs, n)
    for i, (ax, img) in enumerate(zip(axs, imgs)):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{i}')
        ax.set_axis_off()

    plt.savefig(file_name)