import numpy as np
import torch
from torchvision.utils import make_grid
import wandb
from wandb import AlertLevel
from tqdm import tqdm

from .evaluator import Evaluator
from .dataset import transpose_data
from .utils import KLAnnealing, TeacherForcingScheduler, kl_criterion, mse_criterion


class Trainer:
    def __init__(self, model, train_data_loader, test_data_loader, train_cfg, wandb_cfg):
        self.model = model
        self.data_loader = train_data_loader
        self.iterator = iter(self.data_loader)
        self.evaluator = Evaluator(model, test_data_loader)
        self.train_cfg = train_cfg
        self.wandb_cfg = wandb_cfg
        self.init_wandb()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_cfg.lr)
        self.kl_annealer = KLAnnealing(
            train_cfg.kl_anneal_cyclical,
            train_cfg.kl_anneal_ratio,
            train_cfg.kl_anneal_cycle,
            train_cfg.niters
        )
        self.teacher_forcing_scheduler = TeacherForcingScheduler(
            train_cfg.tfr,
            train_cfg.tfr_start_decay_epoch,
            train_cfg.tfr_decay_step,
            train_cfg.tfr_lower_bound
        )

    def init_wandb(self):
        wandb.init(
            project=self.wandb_cfg.project,
            entity=self.wandb_cfg.entity,
            name=self.wandb_cfg.name
        )

    def train(self, n_past, n_future):
        for epoch in range(self.train_cfg.niters):
            self.model.set_train()
            loss, mse, kld = self.train_epoch(epoch, n_past, n_future)

            beta = self.kl_annealer.get_beta()
            tfr = self.teacher_forcing_scheduler.get_tfr()

            result_dict = {
                'loss': loss,
                'mse': mse,
                'kld': kld,
                'beta': beta,
                'tfr': tfr
            }

            if epoch % self.train_cfg.evaluate_interval == 0 or epoch == self.train_cfg.niters - 1:
                psnr, seq = self.evaluator.evaluate(n_past=n_past, n_future=n_future, return_last_seq=True)
                img = make_grid(torch.cat([seq[0], seq[1]]), nrow=n_past + n_future).cpu().numpy()
                video = seq[1].cpu().numpy().astype(np.uint8)
                result_dict['psnr'] = psnr
                result_dict['image'] = wandb.Image(np.transpose(img, (1, 2, 0)))
                result_dict['video'] = wandb.Video(video)

                if self.model.best_psnr < psnr:
                    self.model.best_psnr = psnr
                    self.model.best_epoch = epoch
                    self.model.save_model()
                    wandb.alert(
                        title='Improvement!',
                        text=f'New best psnr: {psnr} in {self.wandb_cfg.name}',
                        level=AlertLevel.INFO
                    )

            wandb.log(result_dict)
            self.kl_annealer.update()
            self.teacher_forcing_scheduler.update()

    def train_epoch(self, epoch, n_past, n_future):
        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        with tqdm(range(self.train_cfg.epoch_size)) as pbar:
            pbar.set_description(f'Epoch {epoch}/{self.train_cfg.niters}')
            for _ in pbar:
                try:
                    train_seq, train_cond = next(self.iterator)
                except StopIteration:
                    self.iterator = iter(self.data_loader)
                    train_seq, train_cond = next(self.iterator)
                loss, mse, kld = self.train_batch(train_seq, train_cond, n_past, n_future)
                epoch_loss += loss
                epoch_mse += mse
                epoch_kld += kld

                pbar.set_postfix(loss=loss, mse=mse, kld=kld)

        return epoch_loss / self.train_cfg.epoch_size, epoch_mse / self.train_cfg.epoch_size, epoch_kld / self.train_cfg.epoch_size

    def train_batch(self, train_seq, train_cond, n_past, n_future):
        seq_len = n_past + n_future
        train_seq, train_cond = transpose_data(train_seq, train_cond)
        self.model.zero_gradients()
        self.model.init_hiddens()

        mse = 0
        kld = 0
        h_seq = [self.model.encode(train_seq[i], train_cond[i]) for i in range(seq_len)]
        for i in range(1, seq_len):
            h_target = h_seq[i][0]
            z_t, mu1, logvar1 = self.model.posterior(h_target)
            mu2 = None
            logvar2 = None
            if self.model.model_cfg.learned_prior:
                z_t, mu2, logvar2 = self.model.prior(h_target)

            if self.model.model_cfg.last_frame_skip or i < n_past or self.teacher_forcing_scheduler.do_teacher_forcing():
                h, skip = h_seq[i-1]
            else:
                result = self.model.encode(x_pred, train_cond[i])
                if self.model.model_cfg.last_frame_skip:
                    h, skip = result
                else:
                    h = result[0]

            h_pred = self.model.frame_predictor(torch.cat([h, z_t, train_cond[i]], 1))
            x_pred = self.model.decode([h_pred, skip], train_cond[i])
            mse += mse_criterion(x_pred, train_seq[i])
            kld += kl_criterion(mu1, logvar1, mu2, logvar2, mu1.size(0))

        beta = self.kl_annealer.get_beta()
        loss = mse + beta * kld
        loss.backward()
        self.optimizer.step()

        return loss.data.cpu().numpy() / seq_len, mse.data.cpu().numpy() / seq_len, kld.data.cpu().numpy() / seq_len
