"""S2A training logic.

Copyright PolyAI Limited.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from data.data_module import DataModule
from modules.s2a_model import Pheme
from modules.vocoder import VocoderType


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--saving_path", type=str, default="./ckpt")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument(
        "--vocoder_type",
        type=str,
        choices=[voc_type.name for voc_type in VocoderType],
        default=VocoderType.SPEECHTOKENIZER.name,
    )
    parser.add_argument("--vocoder_config_path", type=str, default=None)
    parser.add_argument("--vocoder_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--metapath", type=str, nargs="+", help="paths to train metadata", 
        required=True
    )
    parser.add_argument(
        "--val_metapath", type=str, nargs="+", default=[], 
        help="paths to validation metadata",
    )
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--speaker_embedding_dir", type=str, default=None)
    parser.add_argument("--sampledir", type=str, default="./logs")

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=float, default=200)
    parser.add_argument("--max_length", type=int, default=1600)
    parser.add_argument("--train_bucket_size", type=int, default=8192)
    parser.add_argument("--training_step", type=int, default=800000)
    parser.add_argument("--optim_flat_percent", type=float, default=0.0)
    parser.add_argument("--warmup_step", type=int, default=50)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)

    # Architecture
    parser.add_argument("--ffd_size", type=int, default=3072)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--enc_nlayers", type=int, default=6)
    parser.add_argument("--dec_nlayers", type=int, default=6)
    parser.add_argument("--nheads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--depthwise_conv_kernel_size", type=int, default=5)
    parser.add_argument("--aligner_softmax_temp", type=float, default=1.0)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5)
    parser.add_argument("--use_sem_tokens", type=bool, default=True)
    parser.add_argument("--use_spkr_emb", action="store_true")
    parser.add_argument("--use_text_emb", action="store_true")
    parser.add_argument("--only_inference", action="store_true")

    # Dropout
    parser.add_argument("--speaker_embed_dropout", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # Trainer
    parser.add_argument("--val_check_interval", type=int, default=1)
    parser.add_argument("--max_dataset_samples", type=int, default=-1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument(
        "--precision", type=str, choices=["16", "32", "bf16"], default="bf16"
    )
    parser.add_argument("--nworkers", type=int, default=16)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"],
        default="gpu",
    )
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Data
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_codes", type=int, default=1024)
    parser.add_argument("--n_cluster_groups", type=int, default=7)
    parser.add_argument("--first_n_lvls", type=int, default=7)
    parser.add_argument("--use_pretrained_ckpt_cfg", action="store_true")
    parser.add_argument("--n_semantic_codes", type=int, default=1024)

    # Distribution
    parser.add_argument("--sagemaker", action="store_true")

    args = parser.parse_args()

    return args


def split_metapath(in_paths: List[str]):
    podidx_paths, other_paths = [], []

    for itm_path in in_paths:
        if itm_path.endswith("jsonl"):
            podidx_paths.append(itm_path)
        else:
            other_paths.append(itm_path)

    return podidx_paths, other_paths


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.saving_path, exist_ok=True)

    with open(os.path.join(args.saving_path, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.pretrained_path:
        if (
            Path(args.pretrained_path).with_name("config.json").exists()
            and args.use_pretrained_ckpt_cfg
        ):
            with open(
                Path(args.pretrained_path).with_name("config.json"), "r") as f:
                prev_cfg = json.load(f)
            for k, v in prev_cfg.items():
                if isinstance(v, (int,)):
                    if args.__dict__[k] != v:
                        print(f"updating {k}!", args.__dict__[k], v)
                    args.__dict__[k] = v

    fname_prefix = f""
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.saving_path,
        filename=(fname_prefix + "{epoch}-{step}"),
        every_n_train_steps=(
            None if args.val_check_interval == 1.0 else args.val_check_interval  # noqa
        ),
        every_n_epochs=(
            None if args.check_val_every_n_epoch == 1 else args.check_val_every_n_epoch  # noqa
        ),
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor="val/dataset_0/acc_top_5",
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger_tb = TensorBoardLogger(
        args.saving_path, name="VQ-TTS", version=args.version)
    logger_wandb = WandbLogger(project="mqtts", log_model=True, config=args)

    distribution_kwargs = {
        "accelerator": "gpu",
        "strategy": "ddp_find_unused_parameters_true" if args.distributed else "auto", # noqa
    }

    wrapper = Trainer(
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        num_sanity_val_steps=20,
        max_steps=args.training_step,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=[logger_tb, logger_wandb],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        profiler="simple",
        use_distributed_sampler=False,
        # distribution
        **distribution_kwargs,
    )
    model = Pheme(args)
    logger_wandb.watch(model=model)
    _, other_metapath = split_metapath(args.metapath)
    _, other_val_metapath = split_metapath(args.val_metapath)

    print(
        f"Received datasets: \n{other_metapath = } "
        f"\n \n{other_val_metapath = }"
    )

    other_meta = {}
    if len(other_metapath) > 0:
        other_meta["fit"] = other_metapath
    if len(other_val_metapath) > 0:
        other_meta["valid"] = other_val_metapath

    data_module = DataModule(
        args, other_metapath, other_val_metapath,
        wrapper.world_size, wrapper.local_rank
    )
    data_module.setup(stage="fit")
    train_data_module = data_module

    valid_dataloaders = []
    data_module.setup(stage="valid")
    valid_dataloaders.extend(data_module.val_dataloader())

    wrapper.fit(
        model,
        train_dataloaders=train_data_module.train_dataloader(),
        val_dataloaders=valid_dataloaders,
        ckpt_path=args.resume_checkpoint,
    )
