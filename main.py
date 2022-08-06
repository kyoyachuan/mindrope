import random

import hydra
from omegaconf import DictConfig
import torch

from mindrope.dataset import get_dataloader
from mindrope.svg import SVGModel
from mindrope.trainer import Trainer


torch.backends.cudnn.benchmark = True


@hydra.main(config_path='configs', config_name='')
def main(cfg: DictConfig) -> None:
    random.seed(cfg.base.seed)
    torch.manual_seed(cfg.base.seed)
    torch.cuda.manual_seed_all(cfg.base.seed)

    train_dataloader = get_dataloader(cfg.data, mode='train', num_workers=cfg.base.num_workers)
    validate_dataloader = get_dataloader(cfg.data, mode='validate', num_workers=cfg.base.num_workers)
    model = SVGModel(cfg.model, batch_size=cfg.data.batch_size)
    trainer = Trainer(model, train_dataloader, validate_dataloader, cfg.trainer, cfg.wandb)
    trainer.train(cfg.data.n_past, cfg.data.n_future)


if __name__ == '__main__':
    main()
