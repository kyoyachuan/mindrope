import numpy as np
import torch

from .svg import SVGModel
from .dataset import transpose_data
from .utils import finn_eval_seq


class Evaluator:
    def __init__(self, model: SVGModel, data_loader: torch.utils.data.DataLoader):
        self.model = model
        self.data_loader = data_loader
        self.iterator = iter(self.data_loader)

    @torch.no_grad()
    def evaluate(self, n_past, n_future, return_last_seq=False):
        self.model.set_evaluate()
        psnr_list = []
        for _ in range(len(self.data_loader)):
            try:
                validate_seq, validate_cond = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.data_loader)
                validate_seq, validate_cond = next(self.iterator)
            validate_seq, validate_cond = transpose_data(validate_seq, validate_cond)
            pred_seq = self.model.predict_sequence(validate_seq, validate_cond, n_past, n_future)
            _, _, psnr = finn_eval_seq(validate_seq[n_past:], pred_seq[n_past:])
            psnr_list.append(psnr)

        if return_last_seq:
            last_gt = torch.squeeze(torch.transpose(validate_seq, 0, 1)[-1])
            last_pred = torch.squeeze(torch.transpose(pred_seq, 0, 1)[-1])
            return np.mean(np.concatenate(psnr_list)), (last_gt, last_pred)
        return np.mean(np.concatenate(psnr_list))
