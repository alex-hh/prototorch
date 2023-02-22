import os
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
import torch


def calc_grad_norm(params):
    grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]), 2
    )

    return grad_norm


class BaseLearner:
    """Base learner class.

    A learner must implement a train_step method that performs
    parameter updates given a batch of data
    """

    def __init__(
        self,
        model,
        optimizer,
        device=None,  # in a distributed setting we might not want to set this
        lr_scheduler=None,
        max_grad_norm=None,
        accelerate=False,
        accumulation_steps=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        if accelerate:
            # https://huggingface.co/docs/accelerate/basic_tutorials/migration
            # more fine-grained guide to migration here:
            # https://huggingface.co/docs/accelerate/v0.7.1/en/accelerator#accelerate.Accelerator.print
            kwargs_objs = [
                # https://github.com/huggingface/accelerate/issues/314
                # https://huggingface.co/docs/accelerate/package_reference/kwargs
                InitProcessGroupKwargs(timeout=timedelta(seconds=1800*3))
            ]
            self.accelerator = Accelerator(gradient_accumulation_steps=self.accumulation_steps, kwargs_handlers=kwargs_objs)
            # Q. is it ok to call prepare once here for these things, and then later on
            # for data loader? YES: see https://huggingface.co/docs/accelerate/quicktour
            msg = ""
            for ix, (name, param) in enumerate(self.model.named_parameters()):
                msg += f"{self.accelerator.process_index}, {ix}, {name}, {param.shape}, {param.dtype}, {param.device}\n"
            self.accelerator.wait_for_everyone()
            print(msg)

            self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
            if self.accelerator.is_main_process:
                print("accelerator device", self.accelerator.device, flush=True)
            self.device = self.accelerator.device
        else:
            assert self.device is not None
            print("Device", self.device, flush=True)
            self.accelerator = None
            self.model.to(self.device)

    def set_state(self, mode):
        if hasattr(self, "model"):
            if mode == "train":
                self.model.train()
            elif mode == "eval":
                self.model.eval()
            else:
                raise ValueError(f"mode must be train or eval but got {mode}")

    def epoch_begin(self):
        """Called at the start of each training epoch."""
        self.set_state("train")
        self._epoch_step = 0

    def epoch_end(self):
        pass

    def train_begin(self):
        """Called at the start of a training loop."""
        # TODO add a reload option + method
        self._input_step = 0
        if not hasattr(self, "_update_step"):
            self._update_step = 0

    def train_end(self):
        pass

    def test_begin(self):
        """Called at the start of a single pass through val/test data."""
        self.set_state("eval")

    def __call__(self, batch):
        """Call forward_step on batch."""
        return self.forward_step(batch)

    def forward_step(self, batch, is_train=False):
        """Run a forward pass, computing loss and metrics on input batch.

        (loss is a tensor scalar and metrics a dict of python scalars)
        """
        raise NotImplementedError()

    def get_batch_size(self, batch):
        """Compute batch size (helps in metric accumulation)."""
        raise NotImplementedError()

    def accelerated_train_step(self, batch):
        """
        Model is duplicated across all processes, so in EACH PROCESS
        we perform optimizer steps.

        Accelerator handles the synchronisation of gradients and of optimizer
        state to allow this to happen seamlessly.

        Metrics container is per-process. Ultimately we will perform a reduce
        to get a metrics container across all processes.
        """
        # c.f. optimizer_state_was_skipped for gradient skipping
        with self.accelerator.accumulate(self.model):
            # n.b. this handles end of data loader 'overflow' batches automatically
            # (I used to handle this via epoch_end)
            # if self.accelerator.is_main_process:
            # print("Forward", self.summarise_batch_for_debugging(batch), flush=True)
            loss, metrics = self.forward_step(batch, is_train=True)
            self.accelerator.backward(loss)
            # Q why is the sync_gradients condition necessary
            # I think because of gradient accumulation: if we aren't accumulating
            # this batch, we aren't synchronising.
            # https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization
            if self.accelerator.sync_gradients:
                metrics["grad_norm"] = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._update_step += 1
                self._epoch_step += 1

            self._input_step += 1
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                metrics["lr"] = torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device)

            self.optimizer.zero_grad()

        return metrics

    def train_step(self, batch):
        if self.accelerator is not None:
            return self.accelerated_train_step(batch)

        else:
            try:
                loss, metrics = self.forward_step(batch, is_train=True)
                loss = loss / self.accumulation_steps
                loss.backward()

                if (self._epoch_step + 1) % self.accumulation_steps == 0:
                    if self.max_grad_norm is not None:
                        metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
                        metrics["clipped_grad_norm"] = calc_grad_norm(self.model.parameters())

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._update_step += 1

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                        metrics["lr"] = torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device)

                self._input_step += 1
                self._epoch_step += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    # c.f. notes on oom in this repo
                    # c.f. https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/16
                    # https://discuss.pytorch.org/t/is-it-safe-to-recover-from-cuda-oom/12754/3
                    # https://github.com/facebookresearch/fairseq/blob/50a671f78d0c8de0392f924180db72ac9b41b801/fairseq/trainer.py#L188 # noqa: E501
                    torch.cuda.empty_cache()  # may not be necessary (see fairseq and links)
                    print("Failed on batch", self.summarise_batch_for_debugging(batch), "skipping", flush=True)
                else:
                    print("Failed on batch", self.summarise_batch_for_debugging(batch))
                    raise e

            return metrics

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def test_step(self, batch):
        """Compute metrics on a val/test batch."""
        loss, metrics = self.forward_step(batch, is_train=False)

        return metrics

    def _checkpoint_accelerated(self, output_dir, epoch, start_epoch=0):
        # https://huggingface.co/docs/accelerate/quicktour#savingloading-a-model
        # https://huggingface.co/docs/accelerate/quicktour#savingloading-entire-states
        self.accelerator.wait_for_everyone()
        # unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            # saves files within output_dir/checkpoint
            self.accelerator.save_state(os.path.join(output_dir, "checkpoint"))

    def _load_accelerator_checkpoint(self, output_dir):
        self.accelerator.load_state(output_dir)

    def checkpoint(self, output_dir, epoch, start_epoch=0):
        """Save model weights (and optimizer/scheduler state) to file."""
        if self.accelerator is not None:
            self._checkpoint_accelerated(output_dir, epoch, start_epoch=start_epoch)
        else:
            file_ext = f".{'' if start_epoch == 0 else (str(start_epoch) + '.')}pt"
            os.makedirs(
                output_dir, exist_ok=True
            )  # needed b.c. can occur before save_config during first epoch.

            d = {
                "epoch": epoch,
                "weights": self.state_dict(),
                "step": self._update_step,
            }

            d["optimizer"] = self.optimizer.state_dict()
            if self.lr_scheduler is not None:
                d["scheduler"] = self.lr_scheduler.state_dict()
            torch.save(d, os.path.join(output_dir, "checkpoint" + file_ext))

    def load_checkpoint_from_file(
        self,
        chkpt_file,
        device=None,
    ):
        # TODO handle device ...
        chkpt = torch.load(chkpt_file, map_location=device)
        if "step" in chkpt:
            self._update_step = chkpt["step"]
        self.load_state_dict(chkpt["weights"])
        self.optimizer.load_state_dict(chkpt["optimizer"])
        if self.lr_scheduler is None and "scheduler" in chkpt:
            print("Loaded scheduler state but scheduler must be passed to reinstantiate")
        elif "scheduler" in chkpt:
            print("Loaded scheduler")
            self.lr_scheduler.load_state_dict(chkpt["scheduler"])

    def load_checkpoint(self, output_dir, start_epoch=0):
        """Load model weights (and optimizer state) from file."""
        if self.accelerator is not None:
            self._load_accelerator_checkpoint(self, output_dir)
        else:
            filename = f"checkpoint.{'' if start_epoch == 0 else (str(start_epoch) + '.')}pt"
            chkpt_file = os.path.join(output_dir, filename)
            return self.load_checkpoint_from_file(chkpt_file, device=self.device)
