from torch.utils import data
import numpy as np
import os

def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    imgs_masks_paths = []
    data_path = os.path.join(dir, 'data')
    seg_masks_path = os.path.join(dir, 'seg_masks')

    data_list = sorted(os.listdir(data_path))

    for i in data_list:
        item = (os.path.join(data_path, i), os.path.join(seg_masks_path, i))
        imgs_masks_paths.append(item)

    return imgs_masks_paths

class MRI(data.Dataset):
    def __init__(self, img_folder, max_value, min_value, transform=None, target_transform=None):
        self.imgs_masks_paths = make_dataset(img_folder)
        if len(self.imgs_masks_paths) == 0:
            raise (RuntimeError("Found 0 images in: " + img_folder))

        self.transform = transform
        self.target_transform = target_transform
        self.prev_max = max_value
        self.prev_min = min_value
        self.new_max = 1
        self.new_min = 0

    def __getitem__(self, index):
        imgs_masks_path = self.imgs_masks_paths[index]
        pat_id = imgs_masks_path[0].split('/')[-1]
        img = np.load(imgs_masks_path[0])
        seg_mask = np.load(imgs_masks_path[1])

        # normalize in range [0, 1]
        prev_range = self.prev_max - self.prev_min
        new_range = self.new_max - self.new_min
        img = (((img - self.prev_min) * new_range) / prev_range) + self.new_min

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            seg_mask = self.target_transform(seg_mask)

        return img, seg_mask, pat_id

    def __len__(self):
        return len(self.imgs_masks_paths)
