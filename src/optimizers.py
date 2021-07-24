import numpy as np
from torch.optim import Adam


class NoamOptimizer(Adam):

    def __init__(self, params, d_model, factor=2, warmup_steps=4000, betas=(0.9, 0.98), eps=1e-9):
        # self.optimizer = Adam(params, betas=betas, eps=eps)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr = 0
        self.step_num = 0
        self.factor = factor

        super(NoamOptimizer, self).__init__(params, lr=0, betas=betas, eps=eps)

    def step(self, closure=None):
        self.step_num += 1
        self.lr = self.lrate()
        for group in self.param_groups:
            group['lr'] = self.lr
        super(NoamOptimizer, self).step()

    def lrate(self):
        return self.factor * self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))


class ScheduledOptim(Adam):
    def __init__(self, params, betas, eps, d_model, n_warmup_steps):
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
        super().__init__(params, lr=self.init_lr, betas=betas, eps=eps)

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        super().step()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self.param_groups:
            param_group['lr'] = lr
