import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
from utils.img_loader import load_img
from scipy.ndimage.interpolation import zoom


class TestDataset(Dataset):
    def __init__(self, x_pths, y_pths):
        self.x_pths = x_pths
        self.y_pths = y_pths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.x_pths)

    def __getitem__(self, idx):
        x_pth = self.x_pths[idx]
        x = Image.fromarray(np.uint8(load_img(x_pth)))
        x = self.transform(x)
        x = x.squeeze(0).numpy()

        y_pth = self.y_pths[idx]
        y = load_img(y_pth)[:, :, 1]
        y = self._remove_artifacts(y)
        y = self._resize(y)
        return x, y

    def _resize(self, img):
        size = 224
        x, y = img.shape
        return zoom(img, (size / x, size / y), order=0)

    def _remove_artifacts(self, mask):
        mask[mask < 240] = 0  # remove artifacts
        mask[mask >= 240] = 255
        return mask
