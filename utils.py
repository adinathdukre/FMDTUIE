import glob
import os

import numpy as np
import torchvision.transforms.functional as T
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(folder, f'*{ext}')) + glob.glob(os.path.join(folder, f'*{ext.upper()}'))
    return sorted(set(files))


class UIEDataset(Dataset):
    """Test-time dataset for underwater image enhancement.

    mode='paired': every file holds the degraded input (left half) and the reference (right half) side by side,
                   as in the sample images under ./datasets/images/. Each half is resized to size x size.
    mode='single': every file is a single degraded underwater image, resized to size x size.

    The network expects square inputs whose side is divisible by 32 (default 512, as in the paper).
    """

    def __init__(self, data_path, mode='paired', size=512):
        super().__init__()
        assert mode in ('paired', 'single'), mode
        assert size % 32 == 0, 'size must be divisible by 32'
        self.mode, self.size = mode, size
        self.images = list_images(data_path)
        if not self.images:
            raise FileNotFoundError(f'No images found in {data_path}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        name = os.path.splitext(os.path.basename(path))[0] + '.png'
        img = Image.open(path).convert('RGB')
        orig_w, orig_h = img.size

        if self.mode == 'paired':
            img = np.array(img.resize((2 * self.size, self.size), Image.BICUBIC))
            inp = T.to_tensor(img[:, :self.size, :])
            gt = T.to_tensor(img[:, self.size:, :])
            orig_w //= 2
        else:
            inp = T.to_tensor(img.resize((self.size, self.size), Image.BICUBIC))
            gt = inp.new_zeros(0)

        return inp, gt, name, orig_h, orig_w
