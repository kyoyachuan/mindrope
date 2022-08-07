import torch

import mindrope.models as models
from .utils import init_weights


class SVGModel:
    def __init__(self, model_cfg: dict, load_model: bool = False, batch_size: int = 100):
        self.model_cfg = model_cfg
        self.batch_size = batch_size
        self.best_psnr = 0.0
        self.best_epoch = 0
        if load_model:
            self.load_model()
        else:
            self.init_model()
        self.to_gpu()

    def save_model(self):
        save_model = {
            'model_cfg': self.model_cfg,
            'batch_size': self.batch_size,
            'best_epoch': self.best_epoch,
            'best_psnr': self.best_psnr,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'frame_predictor': self.frame_predictor,
            'posterior': self.posterior
        }
        if self.model_cfg.learned_prior:
            save_model['prior'] = self.prior
        torch.save(save_model, self.model_cfg.model_path)

    def load_model(self):
        modules = torch.load(self.model_cfg.model_path)
        self.model_cfg = modules['model_cfg']
        self.batch_size = modules['batch_size']
        self.best_epoch = modules['best_epoch']
        self.best_psnr = modules['best_psnr']
        self.encoder = modules['encoder']
        self.decoder = modules['decoder']
        self.frame_predictor = modules['frame_predictor']
        self.posterior = modules['posterior']
        if self.model_cfg.learned_prior:
            self.prior = modules['prior']

    def to_gpu(self):
        self.encoder.cuda()
        self.decoder.cuda()
        self.frame_predictor.cuda()
        self.posterior.cuda()
        if self.model_cfg.learned_prior:
            self.prior.cuda()

    def init_model(self):
        if self.model_cfg.cond_convolution:
            self.encoder = models.CondVGGEncoder(self.model_cfg.g_dim, ncond=7)
            self.decoder = models.CondVGGDecoder(self.model_cfg.g_dim, ncond=7)
        else:
            self.encoder = models.VGGEncoder(self.model_cfg.g_dim)
            self.decoder = models.VGGDecoder(self.model_cfg.g_dim)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

        self.frame_predictor = models.LSTM(
            self.model_cfg.g_dim + self.model_cfg.z_dim + 7,
            self.model_cfg.g_dim,
            self.model_cfg.rnn_size,
            self.model_cfg.predictor_rnn_layers,
            self.batch_size
        )
        self.posterior = models.GaussianLSTM(
            self.model_cfg.g_dim,
            self.model_cfg.z_dim,
            self.model_cfg.rnn_size,
            self.model_cfg.posterior_rnn_layers,
            self.batch_size
        )
        self.frame_predictor.apply(init_weights)
        self.posterior.apply(init_weights)

        if self.model_cfg.learned_prior:
            self.prior = models.GaussianLSTM(
                self.model_cfg.g_dim,
                self.model_cfg.z_dim,
                self.model_cfg.rnn_size,
                self.model_cfg.prior_rnn_layers,
                self.batch_size
            )
            self.prior.apply(init_weights)

    def zero_gradients(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.frame_predictor.zero_grad()
        self.posterior.zero_grad()
        if self.model_cfg.learned_prior:
            self.prior.zero_grad()

    def init_hiddens(self):
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        if self.model_cfg.learned_prior:
            self.prior.hidden = self.prior.init_hidden()

    def set_train(self):
        self.encoder.train()
        self.decoder.train()
        self.frame_predictor.train()
        self.posterior.train()
        if self.model_cfg.learned_prior:
            self.prior.train()

    def set_evaluate(self):
        self.encoder.eval()
        self.decoder.eval()
        self.frame_predictor.eval()
        self.posterior.eval()
        if self.model_cfg.learned_prior:
            self.prior.eval()

    def parameters(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
            list(self.frame_predictor.parameters()) + list(self.posterior.parameters())
        if self.model_cfg.learned_prior:
            params += list(self.prior.parameters())
        return params

    def encode(self, input, cond):
        if self.model_cfg.cond_convolution:
            return self.encoder(input, cond)
        else:
            return self.encoder(input)

    def decode(self, input, cond):
        if self.model_cfg.cond_convolution:
            return self.decoder(input, cond)
        else:
            return self.decoder(input)

    def sample_z(self, input):
        if self.model_cfg.learned_prior:
            z_t, _, _ = self.prior(input)
            return z_t
        else:
            return torch.cuda.FloatTensor(input.size(0), self.model_cfg.z_dim).normal_()

    def predict_sequence(self, input, cond, n_past, n_future):
        self.init_hiddens()

        pred = [input[0]]
        x_in, c_in = input[0], cond[0]
        for i in range(1, n_past + n_future):
            h = self.encode(x_in, c_in)
            if self.model_cfg.last_frame_skip or i < n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < n_past:
                h_target = self.encode(input[i], cond[i])
                h_target = h_target[0].detach()
                z_t, _, _ = self.posterior(h_target)
                self.sample_z(h)
                self.frame_predictor(torch.cat([h, z_t, cond[i]], 1))
                x_in = input[i]
            else:
                z_t = self.sample_z(h)
                out = self.frame_predictor(torch.cat([h, z_t, cond[i]], 1)).detach()
                x_in = self.decode([out, skip], cond[i]).detach()
            c_in = cond[i]
            pred.append(x_in)
        return torch.stack(pred)
