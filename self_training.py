"""
Modified from the run_squad.py of huggingface's transformers repository:
    https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py
"""

import argparse
import logging
import os
import random
import json
import re

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
    squad_convert_examples_to_features,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor

from metrics import compute_predictions, qa_evaluate

logger = logging.getLogger(__name__)
wandb_logging = False


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
    parser.add_argument("--unlabeled_set", default=None, type=str, help="The input unlabeled set.")
    parser.add_argument("--evaluation_set", default=None, type=str, help="The input evaluation set.")
    parser.add_argument("--threads", default=1, type=int, help="multiple threads for converting example to features")
    parser.add_argument("--exp_name", default="", type=str, help="Experiment name, for logging purpose.")
    parser.add_argument("--lang", default="en", type=str, help="The language of datasets.")

    # training
    parser.add_argument("--iterations", default=3, type=int, help="Number of self-training iterations.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=3, type=int, help="Number of training epochs for each iteration.")
    parser.add_argument("--logging_steps", default=500, type=int, help="Log every X updates steps.")
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--prob_threshold", default=None, type=float, required=True, help="Probability threshold for pseudo-labels."
    )
    parser.add_argument("--noise", default=0.0, type=float, help="Level of noise added during predicting pseudo-labels.")
    parser.add_argument(
        "--no_relabel", action="store_true", help="Not to update existing pseudo-labels every iteration."
    )

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
    args.warmup_steps = 0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb
    if args.use_wandb and not (args.username and args.project_name and args.wandb_api_key):
        raise ValueError("Project name and API key must be provided to use wandb logging.")

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def wandb_config(args):
    config = dict(
        start_model=args.start_model,
        batch_size=args.train_batch_size,
        initial_lr=args.learning_rate,
        iterations=args.iterations,
        epochs=args.epochs,
        seed=args.seed,
        threshold=args.prob_threshold,
        noise=args.noise,
    )
    if args.unlabeled_set:
        config["unlabeled_set"] = args.unlabeled_set
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
    logger.setLevel(logging.INFO)
    fmt = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    datefmt = "%m/%d/%Y %H:%M:%S"
    logging.basicConfig(format=fmt, datefmt=datefmt, level=logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, args.exp_name + "_log.txt"))
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(file_handler)


def log(msg, step=None, to_str=False):
    if isinstance(msg, str):
        logger.info(msg)
    elif isinstance(msg, dict):
        if wandb_logging:
            if step:
                wandb.log(msg, step=step)
            else:
                wandb.log(msg)
        if to_str:
            msg_str = ", ".join([f"{k}: {v}" for k, v in msg.items() if k != "custom_step"])
            logger.info(msg_str)


def load_and_cache_examples(args, tokenizer, filepath, mode):
    if mode != "train":
        cached_file = os.path.join(args.output_dir, "cached_{}".format(mode))
        if os.path.exists(cached_file):
            log("Loading cached file {} ...".format(cached_file))
            loaded = torch.load(cached_file)
            features, dataset, examples = (
                loaded["features"],
                loaded["dataset"],
                loaded["examples"],
            )
            return dataset, examples, features

    log("Processing dataset file {} ...".format(filepath))
    processor = SquadV1Processor()
    if mode != "train":
        examples = processor.get_dev_examples(".", filename=filepath)
    else:
        examples = processor.get_train_examples(".", filename=filepath)
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=(mode == "train"),
        return_dataset="pt",
        threads=args.threads,
    )

    if mode != "train":
        log("Saving cached file {} ...".format(cached_file))
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_file)

    return dataset, examples, features


def predict(args, dataloader, features, model, noise=0.0):
    all_results = []
    model.eval()
    for batch in tqdm(dataloader):
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
            feature = features[feature_index.item()]
            unique_id = int(feature.unique_id)
            output = [output[i] for output in outputs.to_tuple()]
            start_logits, end_logits = output
            if noise > 0.0:
                noise_tensor = torch.randn_like(start_logits)
                start_logits = torch.lerp(start_logits, noise_tensor, noise)
                noise_tensor = torch.randn_like(end_logits)
                end_logits = torch.lerp(end_logits, noise_tensor, noise)
            result = SquadResult(unique_id, to_list(start_logits), to_list(end_logits))
            all_results.append(result)
    return all_results


def get_answer_start(example, pred, lang="en"):
    # character
    if lang == "ko" or lang == "zh":
        occurrences = [m.start() for m in re.finditer(re.escape(pred["text"]), example.context_text)]
        if not occurrences:
            return None
        closest = min(occurrences, key=lambda x:abs(x - pred["start_tok_index"]))
        return {"text": pred["text"], "answer_start": closest}

    # word
    def is_whitespace(c):
        return c.isspace() or ord(c) == 0x202F

    prev_is_whitespace = True
    start_word_index = pred["start_word_index"]
    word_count = -1
    for i, c in enumerate(example.context_text):
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                word_count += 1
                if word_count == start_word_index:
                    return {"text": pred["text"], "answer_start": i}
                prev_is_whitespace = False
    return None


def filter_labels(examples, predictions, threshold, lang="en"):
    filtered = {}
    above_thres = 0
    log("Before filtering labels: num_pseudo_label = {}".format(len(predictions)))
    for e in tqdm(examples, desc="Filtering"):
        if e.qas_id not in predictions.keys():
            continue
        pred = predictions[e.qas_id]
        if pred["start_prob"] < threshold or pred["end_prob"] < threshold:
            # print("start_prob: {}, end_prob: {}".format(pred["start_prob"], pred["end_prob"]))
            continue
        if pred["text"] not in e.context_text:
            continue
        above_thres += 1
        ans = get_answer_start(e, pred, lang)
        if ans:
            filtered[e.qas_id] = ans
    log("Filtering labels: num_pseudo_label above threshold = {}".format(above_thres))
    log("After filtering labels: num_pseudo_label = {}".format(len(filtered)))
    return filtered


def output_pseudo_labels(unlabeled_set, pseudo_labels, output):
    with open(unlabeled_set) as reader:
        input_data = json.load(reader)["data"]

    extracted_articles = []
    acc_exact = 0
    acc_text = 0
    for article in input_data:
        extracted_paragraphs = []
        for par in article["paragraphs"]:
            extracted_qas = []
            for qa in par["qas"]:
                qas_id = qa["id"]
                if qas_id in pseudo_labels.keys():
                    ans = pseudo_labels[qas_id]
                    extracted_qas.append({
                        "question": qa["question"],
                        "id": qas_id,
                        "answers": [{"text": ans["text"], "answer_start": ans["answer_start"]}],
                    })
                    text_match_flag = False
                    for ans_gt in qa["answers"]:
                        if ans["text"] == ans_gt["text"]:
                            text_match_flag = True
                            if ans_gt["answer_start"] == ans["answer_start"]:
                                acc_exact += 1
                                break
                    if text_match_flag:
                        acc_text += 1
            if len(extracted_qas) > 0:
                extracted_paragraphs.append({"qas": extracted_qas, "context": par["context"]})
        if len(extracted_paragraphs) > 0:
            extracted_articles.append({"title": article["title"], "paragraphs": extracted_paragraphs})

    with open(output, "w") as writer:
        json.dump({"data": extracted_articles}, writer)
    total = len(pseudo_labels)
    return acc_exact / total, acc_text / total


def label(args, tokenizer, model, labeled_set_path, global_steps, prev_labeled_set_path):
    # TODO: combine prev labeled set if not None
    dataset, examples, features = load_and_cache_examples(
        args,
        tokenizer,
        args.unlabeled_set,
        mode="label",
    )
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.eval_batch_size)
    log({"num_unlabeled_examples": len(examples)}, step=global_steps, to_str=True)

    results = predict(args, dataloader, features, model, args.noise)
    predictions = compute_predictions(
        tokenizer,
        examples,
        features,
        results,
        args.n_best_size,
        args.max_answer_length,
        output_probs=True,
        output_positions=True,
    )

    pseudo_labels = filter_labels(examples, predictions, args.prob_threshold, args.lang)
    if len(pseudo_labels) == 0:
        return None, None
    if prev_labeled_set_path:
        log("Do not relabel")
        # do not relabel
        with open(prev_labeled_set_path) as reader:
            prev = json.load(reader)["data"]
        for article in prev:
            for par in article["paragraphs"]:
                for qa in par["qas"]:
                    qas_id = qa["id"]
                    pseudo_labels[qas_id] = qa["answers"][0]

    return output_pseudo_labels(args.unlabeled_set, pseudo_labels, labeled_set_path)


def train(args, tokenizer, model, labeled_set, global_steps):
    dataset, examples, features = load_and_cache_examples(
        args,
        tokenizer,
        labeled_set,
        mode="train",
    )
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.train_batch_size)
    log({"num_train_examples": len(examples)}, step=global_steps, to_str=True)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(dataloader) * args.epochs),
    )

    model.zero_grad()
    tr_loss = logging_loss = 0.0
    logged_step = global_steps
    for i in range(args.epochs):
        log("Training epoch {}".format(i + 1))
        epoch_iterator = tqdm(dataloader, desc="Step")
        for step, batch in enumerate(epoch_iterator):
            model.train()
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

            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_steps += 1

            if (global_steps % args.logging_steps == 0) or (step + 1 == len(dataloader) and i + 1 == args.epochs):
                log({
                        "loss": (tr_loss - logging_loss) / (global_steps - logged_step),
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=global_steps
                )
                logged_step = global_steps
                logging_loss = tr_loss

    return global_steps


def evaluate(args, tokenizer, model, it, global_steps):
    dataset, examples, features = load_and_cache_examples(
        args,
        tokenizer,
        args.evaluation_set,
        mode="eval",
    )
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.eval_batch_size)
    log({"num_eval_examples": len(examples)}, step=global_steps, to_str=True)
    output_prediction_file = os.path.join(args.output_dir, "prediction_iter_{}.json".format(it))
    output_nbest_file = os.path.join(args.output_dir, "nbest_prediction_iter_{}.json".format(it))

    results = predict(args, dataloader, features, model)
    predictions = compute_predictions(
        tokenizer,
        examples,
        features,
        results,
        args.n_best_size,
        args.max_answer_length,
        output_prediction_file=output_prediction_file,
        output_nbest_file=output_nbest_file,
    )
    results = qa_evaluate(examples, predictions, args.lang)
    log(results, step=global_steps, to_str=True)
    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging_setup(args)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForQuestionAnswering.from_pretrained(args.start_model).to(args.device)
    global_steps = 0

    # self-training
    for i in range(args.iterations):
        log("Self-training iteration {}".format(i + 1))

        log("Labeling...")
        labeled_set_path = os.path.join(args.output_dir, "iter_{}.json".format(i))
        if args.no_relabel and i > 0:
            prev_labeled_set_path = os.path.join(args.output_dir, "iter_{}.json".format(i - 1))
        else:
            prev_labeled_set_path = None
        acc_exact, acc_text = label(args, tokenizer, model, labeled_set_path, global_steps, prev_labeled_set_path)
        if not acc_exact:
            log("No pseudo-labels, abort self-training.")
            break
        log({"pseudo_label_acc_exact": acc_exact, "pseudo_label_acc_text": acc_text}, step=global_steps, to_str=True)

        log("Training...")
        model = BertForQuestionAnswering.from_pretrained(args.start_model).to(args.device)
        global_steps = train(args, tokenizer, model, labeled_set_path, global_steps)

        ckpt_dir = os.path.join(args.output_dir, "checkpoint_iter_{}".format(i + 1))
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        log("Saving checkpoint to {}".format(ckpt_dir))
        model.save_pretrained(ckpt_dir)

        log("Evaluating...")
        evaluate(args, tokenizer, model, i, global_steps)


if __name__ == "__main__":
    main()
