import argparse
import random

import hydra
from omegaconf import DictConfig
import torch

from mindrope.dataset import get_dataloader
from mindrope.svg import SVGModel
from mindrope.evaluator import Evaluator


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", default="configs")
parser.add_argument("--config_name", default="kl_cyclical.yaml")
parser.add_argument("--num_workers", default=6)
parser.add_argument("--seed", default=1)
args = parser.parse_args()


torch.backends.cudnn.benchmark = True


@hydra.main(config_path=args.config_path, config_name=args.config_name)
def main(cfg: DictConfig) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    test_dataloader = get_dataloader(cfg.data, mode='test', num_workers=args.num_workers)
    model = SVGModel(cfg.model, batch_size=cfg.data.batch_size, load_model=True)
    evaluator = Evaluator(model, test_dataloader)
    psnr = evaluator.evaluate(cfg.data.n_past, cfg.data.n_future)
    print(f"Best PSNR: {psnr}")


if __name__ == '__main__':
    main()
