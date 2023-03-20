import numpy as np

class ConstantScheduler(dict):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def __getitem__(self, k):
        return self.c

class LinearScheduler(dict):
    """_summary_
    l(k) = a * k + b
    Args:
        dict (_type_): _description_
    """
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def __getitem__(self, k):
        return self.a * k + self.b

class StepScheduler(dict):
    def __init__(self, bmin, bmax, nsteps, iter_per_step):
        super().__init__()
        self.bmin = bmin
        self.bmax = bmax
        self.nsteps = nsteps
        self.iter_per_step = iter_per_step
        self.bvalues = np.linspace(bmin, bmax, nsteps)

    def __getitem__(self, k):
        return self.bvalues[k // self.iter_per_step]

class State_fn:
    """tune the beta parameter if the loss is on a plateau, and return when the algo should be stopped
    """
    def __init__(self, cfg) -> None:
        track = cfg.exp.scheduler.track
        if not track in ('residual', 'loss'):
            raise ValueError('track option should be set to \"residual\" or \"loss\"')
        self.track = cfg.exp.scheduler.track
        self.patience =cfg.exp.scheduler.patience
        self.tol = cfg.exp.scheduler.tol
        self.historic = []
        self.it_count = 0 # total iteration count
        self.count = 0 # count since last improvement of the loss
        #self.it_count_since_last_beta_update = 0
        self.beta_min = cfg.exp.scheduler.beta_min #TODO: remove for demo?
        self.beta_max = cfg.exp.scheduler.beta_max
        self.delta_beta = cfg.exp.scheduler.delta_beta
        self.current_beta = cfg.exp.scheduler.beta_min
        self.current_min_val = np.inf

    def step(self, loss, residual):
        self.it_count += 1
        cont = True
        if self.track == 'loss':
            self.historic.append(loss)
            val = loss
        else:
            self.historic.append(residual)
            val = residual
        # deal with negative loss
        s = np.sign(self.current_min_val)
        if val < (1-s*self.tol) * self.current_min_val:
            self.current_min_val = val
            self.count = 0
        else:
            self.count += 1
        if self.count > self.patience:
            self.current_beta += self.delta_beta
            self.current_min_val = np.inf
            self.count = 0
        if self.current_beta > self.beta_max + 10 ** -12:
            cont = False
        return self.current_beta, cont