import torch
import numpy as np


class Net:
    def __init__(self, model_pth):
        self._net = torch.load(model_pth)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, x):
        self._net.to(self.device)
        self._net.eval()
        with torch.no_grad():
            xs = torch.from_numpy(np.stack((x, x, x), 0)).unsqueeze(0).float().to(self.device)
            ys = self._net(xs).to("cpu")
            y = ys.squeeze(0).numpy()[1]
        return self._norm_zero_one(y)

    def _norm_zero_one(self, image):
        """Image normalisation. Normalises image to fit [0, 1] range."""
        image = image.astype(np.float32)

        minimum = np.min(image)
        maximum = np.max(image)

        if maximum > minimum:
            ret = (image - minimum) / (maximum - minimum)
        else:
            ret = image * 0.
        return ret
