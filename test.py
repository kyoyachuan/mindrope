import argparse

import hydra
from omegaconf import DictConfig, OmegaConf


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="test.yaml")
args = parser.parse_args()


@hydra.main(config_path='configs', config_name=args.config)
def test(cfg: DictConfig):
    print(cfg)
    print(cfg.data)
    x = dict(cfg.configs)
    print(x)


if __name__ == '__main__':
    test()
