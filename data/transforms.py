import numpy as np
import torch

"""
Data to tensor
"""
class dataToTensor(object):

    def __call__(self, img):
        trans_array = np.array(img.transpose(2, 0, 1), dtype=np.float64)

        return torch.from_numpy(trans_array).float()

"""
Mask to tensor
"""
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()