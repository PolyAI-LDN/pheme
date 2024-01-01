"""Train T2S to generate semantic tokens.

Copyright PolyAI Limited.
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from data.semantic_dataset import Collator, ConcatenateSemanticDataset
from modules.t2s_model import T2S, compute_custom_metrics
from utils import split_metapath


# Synchronize the GPU
torch.cuda.synchronize()

# Check for CUDA errors
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(torch.cuda.get_device_properties(device))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metapath", type=str, nargs="+", help="paths to train metadata", 
        required=True
    )
    parser.add_argument(
        "--val_metapath", type=str, nargs="+", default=[], 
        help="paths to validation metadata",
    )
    parser.add_argument(
        "--train_path", type=str, 
        default="datasets/giga-training-data/train.json"
    )
    parser.add_argument(
        "--eval_path", type=str, 
        default="datasets/giga-training-data/dev.json"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model_size", choices=["test", "tiny", "t5small", "large", "Large"], 
        default="tiny"
    )
    parser.add_argument("--eval_accumulation_steps", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--nworkers", type=int, default=8)
    parser.add_argument("--max_duration", type=int, default=15)
    parser.add_argument("--eval_n_samples", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=5E-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    model = T2S(args)
    n_params = sum([param.numel() for param in model.parameters()])
    print(f"Model has {n_params = }")

    train_path = split_metapath(args.metapath)
    eval_paths = split_metapath(args.val_metapath)

    dataset_train = ConcatenateSemanticDataset(
        manifest_path=train_path,
        symbol_table_path=model.text_tokens_file,
        max_duration=args.max_duration
    )

    dataset_eval = ConcatenateSemanticDataset(
        manifest_path=eval_paths,
        symbol_table_path=model.text_tokens_file,
        n_samples=args.eval_n_samples,
        max_duration=args.max_duration
    )

    current_timestamp = datetime.now()
    current_timestamp = current_timestamp.strftime("%Y-%m-%d-%H:%M:%S")
    if args.resume_from_checkpoint is not None:
        output_dir = Path(args.resume_from_checkpoint).parent
    else:
        output_dir = Path(args.output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.n_epochs,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=args.nworkers,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to=["all"],
        bf16=False,
        warmup_steps=args.warmup_steps,
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=args.eval_accumulation_steps
    )
    
    trainer = Trainer(
        model=model.t2s,
        args=training_args,
        data_collator=Collator().collate,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        compute_metrics=compute_custom_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
