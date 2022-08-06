import random

import hydra
from omegaconf import DictConfig
import torch

from mindrope.dataset import get_dataloader
from mindrope.svg import SVGModel
from mindrope.evaluator import Evaluator


torch.backends.cudnn.benchmark = True


@hydra.main(config_path='configs', config_name='')
def main(cfg: DictConfig) -> None:
    random.seed(cfg.base.seed)
    torch.manual_seed(cfg.base.seed)
    torch.cuda.manual_seed_all(cfg.base.seed)

    test_dataloader = get_dataloader(cfg.data, mode='test', num_workers=cfg.base.num_workers)
    model = SVGModel(cfg.model, batch_size=cfg.data.batch_size, load_model=True)
    evaluator = Evaluator(model, test_dataloader)
    psnr = evaluator.evaluate(cfg.data.n_past, cfg.data.n_future)
    print(f"Best PSNR: {psnr}")


if __name__ == '__main__':
    main()
