# Improving Cross-Lingual Reading Comprehension with Self-Training

Original implementation of the paper:

Wei-Cheng Huang, Chien-yu Huang, and Hung-yi Lee. 2021. Improving Cross-Lingual Reading Comprehension with Self-Training.


# Instructions to reproduce our experiments

Our method consists of two stages: the fine-tuning stage and the self-training stage,
which corresponds to the two scripts "finetune.py" and "self_training.py".
To replicate our experiments,
first run the "finetune.py" script to fine-tune the pre-trained model on the source language training set,
then run the "self_training.py" script to self-train the model on the unlabeled set (in target language).

## Installation

To run our scripts, the following modules are required:

* pytorch
* numpy
* transformers
* wandb
* tqdm

## Fine-Tuning

### Parameters

`--start_model`: the model to be fine-tuned,
can be a pre-trained checkpoint identifier or a directory containing the model to be used.
In our experiments we always use "bert-base-multilingual-cased".

`--output_dir`: the directory to store the log, predictions, and fine-tuned model.

`--training_set`: the path to the training set file (json).

`--evaluation_set`: the path to the evaluation set file (json).

`--threads` (optional): default 1, threads used to process the dataset,
larger values like 4 or 8 are recommended to speed up the process.

`--exp_name` (optional): name of the experiment,
the logs will be write to `[exp_name]_log.txt` under the output directory.

`--eval_lang`: the language of the evaluation set, used in calculating F1 scores.
possible values: `"en"`, `"fr"`, `"ko"`, `"zh"`.

`--train_batch_size` (optional): default 8, batch size for training.

`--epochs` (optional): default 3, number of epochs to fine-tune.

`--seed` (optional): default 42, the random seed.

### Example

The following example fine-tunes the pre-trained m-BERT using the training set of SQuAD,
and evaluate on the dev set of SQuAD:

```
python finetune.py \
    --start_model bert-base-multilingual-cased --output_dir mbert-squad \
    --training_set train-v1.1.json --evaluation_set dev-v1.1.json \
    --threads 8 --eval_lang en
```

## Self-Training

### Parameters

`--start_model`: the model to be fine-tuned,
can be a pre-trained checkpoint identifier or a directory containing the model to be used.
In our experiments we input the directory path containing the fine-tuned model.

`--output_dir`: the directory to store the log, predictions, and fine-tuned model.

`--unlabeled_set`: the path to the unlabeled set, e.g. the target language training set.

`--evaluation_set`: the path to the evaluation set, e.g. the target language dev set.

`--threads` (optional): default 1, threads used to process the dataset,
larger values like 4 or 8 are recommended to speed up the process.

`--exp_name` (optional): name of the experiment,
the logs will be write to `[exp_name]_log.txt` under the output directory.

`--lang`: the language of the evaluation set, used in calculating F1 scores.
possible values: `"en"`, `"fr"`, `"ko"`, `"zh"`.

`--iterations` (optional): default 3, number of self-training iterations.

`--train_batch_size` (optional): default 8, batch size for training.

`--epochs` (optional): default 3, number of training epochs in each self-training iteration.

`--seed` (optional): default 42, the random seed.

`--prob_threshold`: the probability threshold to filter the pseudo-labels, requires a float between 0 and 1.

### Example

The following examples use the model fine-tuned on SQuAD (stored under the directory `mbert-squad`),
and self-train it on DRCD:

```
python self_training.py \
    --start_model mbert-squad --output_dir squad-drcd \
    --unlabeled_set DRCD_training.json --evaluation_set DRCD_dev.json \
    --threads 8 --lang zh --prob_threshold 0.7
```
