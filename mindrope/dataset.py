import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


default_transform = transforms.Compose([
    transforms.ToTensor(),
])


def transpose_data(x, cond):
    actions = cond[0]
    actions = actions.clone().detach().float().cuda()
    actions = torch.transpose(actions, 0, 1)
    positions = cond[1]
    positions = positions.clone().detach().float().cuda()
    positions = torch.transpose(positions, 0, 1)

    x = x.cuda()
    x = torch.transpose(x, 0, 1)
    return x, torch.cat([actions, positions], -1)


class BairDataset(Dataset):
    def __init__(self, cfg, mode='train', transform=default_transform, max_seq_len=30):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        self.root = '{}/{}'.format(cfg.data_root, mode)
        self.seq_len = cfg.n_past + cfg.n_future
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.sliding_window = cfg.sliding_window
        if mode == 'train':
            self.ordered = False
        else:
            self.ordered = True

        self.transform = transform
        self.dirs = []
        for dir1 in os.listdir(self.root):
            for dir2 in os.listdir(os.path.join(self.root, dir1)):
                self.dirs.append(os.path.join(self.root, dir1, dir2))

        self.seed_is_set = False
        self.idx = 0
        self.cur_dir = self.dirs[0]

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return len(self.dirs)

    def get_seq(self, start_idx):
        if self.ordered:
            self.cur_dir = self.dirs[self.idx]
            if self.idx == len(self.dirs) - 1:
                self.idx = 0
            else:
                self.idx += 1
        else:
            self.cur_dir = self.dirs[np.random.randint(len(self.dirs))]

        image_seq = []
        for i in range(start_idx, self.seq_len + start_idx):
            fname = '{}/{}.png'.format(self.cur_dir, i)
            img = Image.open(fname)
            image_seq.append(self.transform(img))
        image_seq = torch.stack(image_seq)

        return image_seq

    def get_csv(self, start_idx):
        with open('{}/actions.csv'.format(self.cur_dir), newline='') as csvfile:
            rows = csv.reader(csvfile)
            actions = []
            for i, row in enumerate(rows):
                if i < start_idx:
                    pass
                if i == self.seq_len:
                    break
                action = [float(value) for value in row]
                actions.append(torch.tensor(action))

            actions = torch.stack(actions)

        with open('{}/endeffector_positions.csv'.format(self.cur_dir), newline='') as csvfile:
            rows = csv.reader(csvfile)
            positions = []
            for i, row in enumerate(rows):
                if i < start_idx:
                    pass
                if i == self.seq_len:
                    break
                position = [float(value) for value in row]
                positions.append(torch.tensor(position))
            positions = torch.stack(positions)

        return (actions, positions)

    def get_random_start_index(self):
        return np.random.randint(0, self.max_seq_len - self.seq_len)

    def __getitem__(self, index):
        self.set_seed(index)
        start_idx = 0
        if self.sliding_window:
            start_idx = self.get_random_start_index()
        seq = self.get_seq(start_idx)
        cond = self.get_csv(start_idx)
        return seq, cond


def get_dataloader(cfg: dict, mode='train', num_workers=4):
    assert mode == 'train' or mode == 'test' or mode == 'validate'
    dataset = BairDataset(cfg, mode)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True if mode == 'train' else False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
