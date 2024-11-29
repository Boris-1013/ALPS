from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    """
    A custom learning rate scheduler for warm-up phase.
    Gradually increases the learning rate from 0 to the base learning rate over a specified number of iterations.
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        #Initializes the WarmUpLR scheduler.
        self.total_iters = total_iters  # Total iterations for the warm-up phase
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        #Computes the learning rate for the current iteration.

        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class downLR(_LRScheduler):
    """
    A custom learning rate scheduler for down-scaling phase.
    Gradually decreases the learning rate from the base learning rate to 0 over a specified number of iterations.
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        #Initializes the downLR scheduler.
        self.total_iters = total_iters  # Total iterations for the down-scaling phase
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        #Computes the learning rate for the current iteration.
        return [base_lr * (self.total_iters - self.last_epoch) / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
