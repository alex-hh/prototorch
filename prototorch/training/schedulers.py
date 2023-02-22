from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_lr, verbose=False):
        """see InverseSquareRootScheduler for further description.
        This is just linear warmup with no subsequent decay."""
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        super().__init__(optimizer, verbose=verbose)  # question what happens at init?

    def get_lr(self):
        assert len(self.optimizer.param_groups) == 1
        if self._step_count < self.warmup_steps:
            lr = (self._step_count / self.warmup_steps) * self.max_lr
        else:
            lr = self.max_lr
        return [lr]


class InverseSquareRootScheduler(_LRScheduler):

    """
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py
    https://arxiv.org/abs/1706.03762

    MSA Transformer

    We train 100M parameters model with 12 layers, 768 embedding size
    and 12 attention heads, using a batch size of 512 MSAs, learning rate
    10−4, no weight decay, and an inverse square root learning rate schedule
    with 16000 warmup steps

    ESM

    The model was optimized using Adam (β1 = 0.9, β2 = 0.999) with learning
    rate 10−4. We trained with 131,072 tokens per batch (128 gpus x 1024 tokens).
    The models follow a warm-up period of 16000 updates, during which the
    learning rate increases linearly. Afterwards, the learning rate follows
    an inverse square root decay schedule.
    """

    def __init__(self, optimizer, warmup_steps, max_lr, verbose=False):
        """
        N.B. _LRScheduler has a last_epoch kwarg to enable resuming from checkpoint.
        not sure how this would be compatible with a scheduler that makes a step
        every batch rather than every epoch

        Question how does _step_count and last_epoch differ?

        Warmup steps typically several thousand (4K for transformer, n.b. that
                                                 this is a fraction of epoch size for that model.)
        max lr around 0.0003 - 0.001 depending on model and batch size.
        """
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        super().__init__(optimizer, verbose=verbose)  # question what happens at init?

    def get_lr(self):
        """
        Formulas presented elsewhere e.g. in paper are very counter intuitive.
        to map to paper: lr = d**-.5 min s**-.5, s/(N**1.5)

        when s = N, lr = d**-.5 N**-.5
        when s = 2N, lr = d**-.5 (2N)**-.5 = d**-.5 N**-.5 * (2N/N)**-.5 = max lr * (s/N)**-.5

        max_lr = (d*N)**-.5 , N is N warmup steps

        when step num = warmup_steps

        then during warmup lr = s/N * (d*N)**-.5
             during cooldown lr = 1/root(s/N) * d**-.5
        """
        assert len(self.optimizer.param_groups) == 1
        if self._step_count < self.warmup_steps:
            lr = (self._step_count / self.warmup_steps) * self.max_lr
        else:
            lr = self.max_lr * (self._step_count / self.warmup_steps) ** -0.5
        return [lr]


# class LinearWarmup:

#     """
#     https://github.com/google-research/bert/blob/ffbda2a1aafe530525212d13194cc84d92ed0313/optimization.py#L29-L65

#     This corresponds to increasing the learning rate linearly for the first
#     warmup_steps training steps, and decreasing it thereafter proportionally
#     to the inverse square root of the step number. We used
#     warmup_steps = 4000.
#     """

#     def __init__(self, steps, target_lr):
#         self._step = 0

#     def step(self):
#         return min((steps/self._step)*self.target_lr, self.target_lr)
