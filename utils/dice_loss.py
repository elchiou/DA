import torch.nn.functional as F
import torch.nn as nn

class DiceLoss(nn.Module):
    """Computes the Sørensen–Dice loss.
        Args:
            ignore_index:  the class index to ignore
        Shape:
            outp: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output of the model.
            target: a tensor of shape [B, H, W]. Corresponds to
                the segmentation mask
            eps: added to the denominator for numerical stability.
        Returns:
            loss: the Sørensen–Dice loss.
            scores: a tensor of shape [B, C, H, W]. Corresponds to
                the logits.
        """
    def __init__(self, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outp, target, eps=0.00001):
        num = target.size(0)
        scores = F.softmax(outp)
        encoded_target = outp.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target = target.clone()
            scores = scores.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
            scores[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        intersection = encoded_target * scores
        union = encoded_target * encoded_target + scores * scores
        loss = 2. * intersection[:, 1, :, :].sum(1).sum(1) / (union[:, 1, :, :].sum(1).sum(1) + eps)
        loss = -1. * (loss.sum() / num)

        return loss, scores

