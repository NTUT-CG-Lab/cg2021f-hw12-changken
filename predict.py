import os

from PIL import Image
from tqdm import tqdm

from datasets.test_dataset import TestDataset
from utils.img_loader import rescale_img, threshold_img, write_pred_mask, write_pred_label_mask
from nets.net import Net


class PredDir:
    def __init__(self, root_pred_dir, img_fd):
        self.gt_mask_dir = os.path.join(root_pred_dir, 'gt_masks', img_fd)
        self.pred_mask_dir = os.path.join(root_pred_dir, 'masks', img_fd)
        self.pred_result_dir = os.path.join(root_pred_dir, 'results', img_fd)
        self.pred_comp_results_dir = os.path.join(root_pred_dir, 'comparison_results', img_fd)
        self.mk_dirs()

    def mk_dirs(self):
        dirs = [
            self.gt_mask_dir,
            self.pred_mask_dir,
            self.pred_result_dir,
            self.pred_comp_results_dir
        ]

        # make output dir
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)


def predict(target_id, threshold_ratio=0.9):
    # root output dir
    root_pred_dir = 'data/pred_data'
    os.makedirs(os.path.join(root_pred_dir), exist_ok=True)

    # root src dir
    root_img_dir = 'data/test_data/images'
    root_lab_dir = 'data/test_data/masks'

    # image and label dir
    img_fd = target_id
    img_dir = os.path.join(root_img_dir, img_fd)
    lab_dir = os.path.join(root_lab_dir, img_fd)

    # make root output sub dir
    pred_dirs = PredDir(root_pred_dir, img_fd)

    # set data path
    x_fs = sorted(os.listdir(img_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    y_fs = sorted(os.listdir(lab_dir), key=lambda y: int(y.split('_')[-1].split('.')[0]))

    x_pths = [os.path.join(img_dir, x_f) for x_f in x_fs]
    y_pths = [os.path.join(lab_dir, y_f) for y_f in y_fs]

    # load dataset
    ds = TestDataset(x_pths, y_pths)

    # init model
    model_pth = 'model/model.pth'
    print('Load model ...')
    net = Net(model_pth)

    print('Predict ...')
    # get item form dataset
    for i, (img, lab) in enumerate(tqdm(ds)):
        # predict
        pred = net.predict(img)

        # threshold
        th_pred = threshold_img(pred, threshold_ratio)

        # src image
        inp_img = Image.fromarray(rescale_img(img)).convert("L")
        # src label
        lab_img = Image.fromarray(rescale_img(lab)).convert("L")
        # pred imgs
        mask_img = Image.fromarray(rescale_img(th_pred)).convert("L")

        # save imgs
        lab_img.save(os.path.join(pred_dirs.gt_mask_dir, f'{i}.jpg'))
        mask_img.save(os.path.join(pred_dirs.pred_mask_dir, f'{i}.jpg'))
        write_pred_mask(inp_img, mask_img, os.path.join(pred_dirs.pred_result_dir, f'{i}.jpg'))
        write_pred_label_mask(inp_img, mask_img, lab_img, os.path.join(pred_dirs.pred_comp_results_dir, f'{i}.jpg'))


if __name__ == '__main__':
    target_id = 'ID00423637202312137826377'
    predict(target_id, threshold_ratio=0.9)
