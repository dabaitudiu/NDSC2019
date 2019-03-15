import math


class CosineLRScheduler():
    def __init__(self, duration=20, max_update=50, base_lr=1e-3, final_lr=1e-5):
        if max_update < 1 or duration < 1:
            raise ValueError("maximum number of updates and duration must be strictly positive")
        self.duration = duration
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr

    def update(self, num_update):
        if num_update <= self.max_update:
            num_update = num_update % self.duration
            base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                            (1 + math.cos(math.pi * num_update / self.duration)) / 2
        else:
            base_lr = self.final_lr
        return base_lr
