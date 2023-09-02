import torch


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.update_param(p)

    def update_param(self, p):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)

    def update_param(self, p):
        p_update = -self.lr * p.grad
        p.add_(p_update)


class SGDMomentum(Optimizer):
    def __init__(self, params, lr, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.m_t = {p: torch.zeros_like(p.data) for p in self.params}

    def update_param(self, p):
        self.m_t[p] = self.momentum * self.m_t[p] + (1 - self.momentum) * p.grad
        p_update = -self.lr * self.m_t[p]
        p.add_(p_update)


class Adam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.param_step = {p: 0 for p in self.params}
        self.m_t = {p: torch.zeros_like(p.data) for p in self.params}
        self.v_t = {p: torch.zeros_like(p.data) for p in self.params}

    def update_param(self, p):
        self.param_step[p] += 1

        self.m_t[p] = self.beta1 * self.m_t[p] + (1 - self.beta1) * p.grad
        self.v_t[p] = self.beta2 * self.v_t[p] + (1 - self.beta2) * p.grad**2

        bias_correction_m_t = 1 - self.beta1 ** self.param_step[p]
        bias_correction_v_t = 1 - self.beta2 ** self.param_step[p]

        m_hat = self.m_t[p] / bias_correction_m_t
        v_hat = self.v_t[p] / bias_correction_v_t
        lr = self.lr / (torch.sqrt(v_hat) + self.eps)

        p_update = -lr * m_hat

        p.add_(p_update)
