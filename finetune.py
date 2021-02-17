"""
Modified from the run_squad.py of huggingface's transformers repository:
    https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import wandb

from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.data.processors.squad import SquadResult

from metrics import compute_predictions, qa_evaluate
from utils import dict_to_str, get_logger, load_and_cache_examples, remove_cache

logger = logging.getLogger(__name__)
wandb_logging = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def log_metrics(metrics_dict):
    if wandb_logging:
        wandb.log(metrics_dict)
    logger.info(dict_to_str(metrics_dict))


def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--start_model",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--training_set", default=None, type=str, help="The input training file.")
    parser.add_argument("--evaluation_set", default=None, type=str, help="The input evaluation file.")
    parser.add_argument("--threads", default=1, type=int, help="multiple threads for converting example to features")
    parser.add_argument("--exp_name", default="", type=str, help="Experiment name, for logging purpose.")
    parser.add_argument("--eval_lang", default="en", type=str, help="The language of evaluation set.")

    # training
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--logging_steps", default=500, type=int, help="Log every X updates steps.")
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument("--checkpoint", default=False, type=bool, help="Whether to save checkpoint after epochs.")

    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="Turn on wandb logging.")
    parser.add_argument("--username", default=None, type=str, help="Username for wandb.")
    parser.add_argument("--project_name", default=None, type=str, help="Project name for wandb.")
    parser.add_argument("--wandb_api_key", default=None, type=str, help="wandb API key for login.")
    parser.add_argument("--run_group", default=None, type=str, help="wandb run group")
    parser.add_argument("--job_type", default=None, type=str, help="wandb job type")

    args = parser.parse_args()

    # data
    args.max_seq_length = 384
    args.doc_stride = 128
    args.max_query_length = 64
    args.max_answer_length = 30
    args.n_best_size = 20

    # training
    args.model_name = "bert-base-multilingual-cased"
    args.weight_decay = 0.0
    args.adam_epsilon = 1e-8
    args.max_grad_norm = 1.0
    args.warmup_steps = 0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb
    if args.use_wandb and not (args.username and args.project_name and args.wandb_api_key):
        raise ValueError("Project name and API key must be provided to use wandb logging.")

    return args


def wandb_config(args):
    config = dict(
        start_model=args.start_model,
        batch_size=args.train_batch_size,
        initial_lr=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
    )
    if args.training_set:
        config["training_set"] = args.training_set
    if args.evaluation_set:
        config["evaluation_set"] = args.evaluation_set
    return config


def logging_setup(args):
    if args.use_wandb:
        global wandb_logging
        wandb_logging = True
        os.environ["WANDB_ENTITY"] = args.username
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        if args.exp_name:
            os.environ["WANDB_NAME"] = args.exp_name
        if args.run_group:
            os.environ["WANDB_RUN_GROUP"] = args.run_group
        if args.job_type:
            os.environ["WANDB_JOB_TYPE"] = args.job_type
        wandb.login()
        wandb.init(project=args.project_name, config=wandb_config(args))
    global logger
    logger = get_logger(
        logger,
        os.path.join(args.output_dir, args.exp_name + "_log.txt" if args.exp_name else "log.txt")
    )


def train(args, model, tokenizer, dataloader):
    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(dataloader) * args.epochs
    )

    global_step = 0
    training_loss = 0.0
    logging_loss = 0.0
    for i in range(args.epochs):
        logger.debug(f"Training epoch {i + 1}")

        model.train()
        for batch in tqdm(dataloader, desc="Training"):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            training_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            if wandb_logging and global_step % args.logging_steps == 0:
                wandb.log(
                    {
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/loss": (training_loss - logging_loss) / args.logging_steps,
                    },
                    step=global_step,
                )
                logging_loss = training_loss

        if args.evaluation_set:
            results = epoch_end_eval(args, model, tokenizer)
            log_metrics({
                "epoch": i + 1,
                "eval/EM": results["EM"],
                "eval/F1": results["F1"],
                "eval/Precision": results["Precision"],
                "eval/Recall": results["Recall"],
            })

        if args.checkpoint:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{i + 1}")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            logger.debug(f"Saving checkpoint to {ckpt_dir}")
            model.save_pretrained(ckpt_dir)
            torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))


def epoch_end_eval(args, model, tokenizer):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, "eval")
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.eval_batch_size)
    return evaluate(args, model, tokenizer, dataloader, examples, features)


def evaluate(args, model, tokenizer, dataloader, examples, features, output=False):
    all_results = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            feature_indices = batch[3]
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [to_list(output[i]) for output in outputs.to_tuple()]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    if output:
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    else:
        output_prediction_file = None
        output_nbest_file = None

    predictions = compute_predictions(
        tokenizer,
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        output_prediction_file,
        output_nbest_file,
    )
    results = qa_evaluate(examples, predictions, args.eval_lang)
    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging_setup(args)

    # Initialization
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForQuestionAnswering.from_pretrained(args.start_model).to(args.device)

    # Training loop (including evaluation after every epoch if evaluation_set provided)
    if args.training_set:
        dataset, examples, _ = load_and_cache_examples(args, tokenizer, "train")
        dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.train_batch_size)
        log_metrics({
            "train/examples": len(examples),
            "train/steps_per_epoch": len(dataloader),
            "train/total_steps": len(dataloader) * args.epochs
        })
        if wandb_logging:
            wandb.watch(model, log="all", log_freq=args.logging_steps)

        train(args, model, tokenizer, dataloader)
        logger.debug(f"Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    if args.evaluation_set:
        dataset, examples, features = load_and_cache_examples(args, tokenizer, "eval")
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.eval_batch_size)
        log_metrics({
            "eval/examples": len(dataset),
            "eval/steps": len(dataloader),
        })

        if not args.training_set:
            results = evaluate(args, model, tokenizer, dataloader, examples, features, output=True)
            log_metrics({
                "eval/EM": results["EM"],
                "eval/F1": results["F1"],
                "eval/Precision": results["Precision"],
                "eval/Recall": results["Recall"],
            })
    remove_cache(args.output_dir)


if __name__ == "__main__":
    main()
