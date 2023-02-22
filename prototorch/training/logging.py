import argparse
import glob
import json
import os
import time

import wandb
import yaml

from prototorch.training.utils import isnumeric


def save_config(config, filepath, out_format="yaml"):
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as outfile:
        if out_format == "yaml":
            yaml.dump(config, outfile)
        elif out_format == "json":
            json.dump(config, outfile, indent=4)


def get_versioned_dir(output_dir, version=None, resume=False):
    """version gets dir for specific version, resume gets dir for last version."""
    if version is None:
        current_versions = glob.glob(os.path.join(output_dir, "version*"))
        if current_versions:
            last_version = max([int(os.path.basename(v).split("_")[1]) for v in current_versions])
            version = last_version if resume else last_version + 1
        else:
            assert not resume, f"Passed resume True but no matching directories in {output_dir}"
            version = 1

    version_dir = os.path.join(output_dir, f"version_{version}")
    return version_dir, version


def get_output_dir(
    output_dir,
    output_folder,
    use_versioning=True,
    resume=False,
    create_dir=True,
    version=None,
):
    if output_dir is not None:
        output_dir = os.path.join(output_dir, output_folder)
        if use_versioning:
            output_dir, version = get_versioned_dir(output_dir, version=version, resume=resume)
        if create_dir:
            os.makedirs(output_dir, exist_ok=True)
        if use_versioning:
            return output_dir, version
        else:
            return output_dir

    elif use_versioning:
        return None, None

    else:
        return None


def log_epoch_metrics(epoch, metrics, output_file, extra_keys=None, start_epoch=0, new_file=False):
    """
    New file gets created if epoch == 1

    We are going for a hierarchical structure /experiment_group/model_name/train_metrics.csv etc
    because this works best with tensorboard and avoids file clutter in a single
    experiment_group directory

    tensorboard refs:
        https://pytorch.org/docs/stable/tensorboard.html
        https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
    """
    # output_filename = (model_name + f"_{msa_name}" + f"_vae" +
    # ("_posembed{args.pos_embed_dim}" if args.embed_pos else ""))

    metrics.pop("epoch", None)
    metrics = {k: v for k, v in metrics.items() if isnumeric(v)}
    metric_names = list(metrics.keys())
    extra_keys = extra_keys or []
    assert all([m not in metric_names for m in extra_keys]), f"{metric_names} {extra_keys}"
    metric_names += list(extra_keys)

    if new_file:  # c.f. training/core epoch 0 is for validation.
        with open(output_file, "w") as csvf:
            csvf.write(",".join(["epoch"] + metric_names) + "\n")

    with open(output_file, "a") as csvf:
        csvf.write(
            ",".join([str(epoch + start_epoch)] + [str(metrics.get(m, "")) for m in metric_names])
            + "\n"
        )


class StdOutLogger:
    def __init__(self, log_freq, start_epoch=0):
        self.start_epoch = start_epoch
        self.log_freq = log_freq

    def log(self, epoch, metrics, batch=None, force=False):
        if force or epoch % self.log_freq == 0:
            if batch is None:
                header = f"Epoch {epoch + self.start_epoch}:   "
            else:
                header = f"[{epoch:d}, {batch:5d}]:   "

            train_metric_components = [
                f"{m}: {v:.3f} "
                for m, v in metrics.items()
                if isnumeric(v) and not m.startswith("val/")
            ]
            if train_metric_components:
                print(
                    header + "  ".join(train_metric_components),
                    flush=True,
                )
            val_metric_components = [
                f"{m}: {v:.3f} "
                for m, v in metrics.items()
                if isnumeric(v) and m.startswith("val/")
            ]
            if val_metric_components:
                print(
                    "  ".join(val_metric_components),
                    flush=True,
                )
            if batch is None:
                print("--------------------------------------\n")

    def end(self):
        pass


class CSVLogger:
    def __init__(self, output_dir, start_epoch=0):
        self.output_dir = output_dir
        self.start_epoch = start_epoch
        self.val_keys = None
        self.filename = f"train_log.{'' if start_epoch == 0 else (str(start_epoch) + '.')}csv"
        self.logged = 0

    @property
    def filepath(self):
        return str(os.path.join(self.output_dir, self.filename))

    def log(self, epoch, metrics, batch=None, force=False):
        metrics["batch"] = batch

        if epoch == 1:
            self.val_keys = {k: v for k, v in metrics.items() if isnumeric(v)}
        if self.output_dir is not None and epoch > 0:
            # print([k for k in metrics.keys() if k not in self._prev_keys])
            extra_keys = [k for k in self.val_keys if k not in metrics and k != "epoch"]
            os.makedirs(self.output_dir, exist_ok=True)
            log_epoch_metrics(
                epoch,
                metrics,
                self.filepath,
                extra_keys=extra_keys,
                start_epoch=self.start_epoch,
                new_file=self.logged == 0,
            )
            self.logged += 1
            # self._prev_keys = metrics.keys()

    def end(self):
        pass


class NullWandb:
    def log(self, metrics):
        pass

    def finish(self):
        pass


class WandbLogger:
    def __init__(self, project_name, config, entity, version=None, raise_connection_error=True):
        self.project_name = project_name
        self.config = config
        self.entity = entity
        self.wandb_run = None
        self.version = version
        self.raise_connection_error = raise_connection_error

    def log(self, epoch, metrics, batch=None, force=False):
        run_name = self.config["experiment_name"] + "_" + str(time.time()).split(".")[0]
        if self.version is not None:
            run_name += f"_v{self.version}"

        if self.wandb_run is None:
            try:
                self.wandb_run = wandb.init(
                    project=self.project_name,
                    entity=self.entity,
                    config=self.config,
                    name=run_name,
                    # resume=self.final_epoch>0,
                    # required to ensure multiple inits in same process don't all get logged to
                    # single run.
                    reinit=True,
                )
            except Exception as e:
                if self.raise_connection_error:
                    # we want this on for proper job runs.
                    raise e
                else:
                    print("failed to init wandb")
                    self.wandb_run = NullWandb()

        metrics["epoch"] = epoch
        self.wandb_run.log(metrics)

    def end(self):
        self.wandb_run.finish()


class LoggerContainer:
    def __init__(self, loggers, accelerator=None, start_epoch=0):
        self.train_log = []
        self.accelerator = accelerator
        self.loggers = loggers
        self.start_epoch = start_epoch

    def log(self, epoch, metrics, batch=None, force=False):
        for logger in self.loggers:
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.log(epoch, metrics, batch=batch, force=force)
        metrics["epoch"] = epoch + self.start_epoch
        self.train_log.append(metrics)

    def end(self):
        for logger in self.loggers:
            logger.end()
