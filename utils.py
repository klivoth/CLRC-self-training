import logging
import os

import torch

from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV1Processor


logger = logging.getLogger(__name__)


def dict_to_str(d):
    return ", ".join([f"{k}: {round(v, 2)}" for k, v in d.items()])


def get_logger(logger, log_file_path):
    logger.setLevel(logging.DEBUG)
    msg_fmt = "%(asctime)s - %(levelname)-5s - %(name)s -   %(message)s"
    date_fmt = "%m/%d/%Y %H:%M:%S"

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt=msg_fmt, datefmt=date_fmt)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_and_cache_examples(args, tokenizer, mode, cache=True, filepath=None):
    def get_filepath():
        path = filepath
        if path:
            return path
        if mode == "train":
            path = getattr(args, "training_set", None)
        elif mode == "eval":
            path = args.evaluation_set
        else:
            path = getattr(args, "unlabeled_set", None)

        if not path:
            raise ValueError(f"{mode} set not provided.")
        return path

    def process_dataset(filepath):
        processor = SquadV1Processor()
        if mode == "train":
            examples = processor.get_train_examples(".", filename=filepath)
        else:
            examples = processor.get_dev_examples(".", filename=filepath)
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
        return dataset, examples, features

    if cache:
        cached_path = os.path.join(args.output_dir, f"cached_{mode}")
        if os.path.exists(cached_path):
            logger.debug(f"Loading cached file {cached_path} ...")
            loaded = torch.load(cached_path)
            features, dataset, examples = (
                loaded["features"],
                loaded["dataset"],
                loaded["examples"],
            )
        else:
            dataset, examples, features = process_dataset(get_filepath())
        logger.debug(f"Saving cached file {cached_path} ...")
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_path)
    else:
        dataset, examples, features = process_dataset(get_filepath())

    return dataset, examples, features


def remove_cache(folder):
    files = os.listdir(folder)
    for f in files:
        if not f.startswith("cached_"):
            continue
        cache = os.path.join(folder, f)
        if os.path.isfile(cache):
            os.remove(cache)
