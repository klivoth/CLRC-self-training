"""
Modified from the squad_metrics.py of huggingface's transformers repository:
    https://github.com/huggingface/transformers/blob/master/src/transformers/data/metrics/squad_metrics.py

Evaluation functions for different languages come from the evaluation scripts of the following datasets:
    SQuAD v1.1, FQuAD, KorQuAD
"""

from collections import (
    Counter,
    defaultdict,
    OrderedDict,
    namedtuple,
)
import math
import json
import string
import re
from tqdm import tqdm
import torch
from torch.nn import Softmax
from transformers import BasicTokenizer


def compute_predictions(
    tokenizer,
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    output_prediction_file=None,
    output_nbest_file=None,
    output_probs=False,
    output_positions=False,
):

    example_index_to_features = defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = OrderedDict()
    all_nbest_json = OrderedDict()

    iterator = tqdm(all_examples, desc="Computing prediction")
    for (example_index, example) in enumerate(iterator):
        features = example_index_to_features[example_index]
        best, nbest = compute_predictions_for_example(
            tokenizer,
            example,
            features,
            unique_id_to_result,
            n_best_size,
            max_answer_length,
            output_probs,
            output_positions,
        )
        if not (output_probs or output_positions):
            all_predictions[example.qas_id] = best["text"]
        else:
            all_predictions[example.qas_id] = best
        all_nbest_json[example.qas_id] = nbest

    if output_prediction_file:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    if output_nbest_file:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    return all_predictions


def _get_best_indices(logits, n_best_size):
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indices = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indices.append(index_and_score[i][0])
    return best_indices


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def compute_predictions_for_example(
    tokenizer,
    example,
    features,
    unique_id_to_result,
    n_best_size,
    max_answer_length,
    output_probs,
    output_positions,
):

    prelim_predictions = []
    for (feature_index, feature) in enumerate(features):
        result = unique_id_to_result[feature.unique_id]
        prelim_predictions += compute_predictions_for_feature_prelim(
            feature, feature_index, result, n_best_size, max_answer_length
        )
    prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
    nbest = compute_predictions_for_feature_final(tokenizer, example, features, prelim_predictions, n_best_size)

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry
    probs = _compute_softmax(total_scores)

    nbest_output = []
    for (i, entry) in enumerate(nbest):
        output = OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        if output_probs:
            output["start_prob"] = entry.start_prob
            output["end_prob"] = entry.end_prob
        if output_positions:
            output["start_word_index"] = entry.start_word_index
            output["start_tok_index"] = entry.start_tok_index
        nbest_output.append(output)

    return nbest_output[0], nbest_output


def compute_predictions_for_feature_prelim(feature, feature_index, result, n_best_size, max_answer_length):

    _PrelimPrediction = namedtuple(
        "PrelimPrediction",
        [
            "feature_index",
            "start_index",
            "end_index",
            "start_logit",
            "end_logit",
            "start_prob",
            "end_prob",
        ],
    )
    softmax = Softmax(dim=0)

    prelim_predictions = []
    start_indices = _get_best_indices(result.start_logits, n_best_size)
    end_indices = _get_best_indices(result.end_logits, n_best_size)
    start_probs = softmax(torch.Tensor(result.start_logits))
    end_probs = softmax(torch.Tensor(result.end_logits))

    for start_index in start_indices:
        for end_index in end_indices:
            if start_index >= len(feature.tokens):
                continue
            if end_index >= len(feature.tokens):
                continue
            if start_index not in feature.token_to_orig_map:
                continue
            if end_index not in feature.token_to_orig_map:
                continue
            if not feature.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index],
                    start_prob=start_probs[start_index],
                    end_prob=end_probs[end_index],
                )
            )
    return prelim_predictions


def compute_predictions_for_feature_final(tokenizer, example, features, prelim_predictions, n_best_size):

    _NbestPrediction = namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction",
        ["text", "start_logit", "end_logit", "start_prob", "end_prob", "start_word_index", "start_tok_index"],
    )
    seen_predictions = {}
    nbest = []

    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.feature_index]
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]

            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, False)
            if final_text in seen_predictions:
                continue

            nbest.append(_NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit,
                start_prob=pred.start_prob,
                end_prob=pred.end_prob,
                start_word_index=orig_doc_start,
                start_tok_index=pred.start_index,
            ))

        else:
            final_text = ""
            nbest.append(_NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit,
                start_prob=pred.start_prob,
                end_prob=pred.end_prob,
                start_word_index=0,
                start_tok_index=0,
            ))

        seen_predictions[final_text] = True

    if not nbest:
        nbest.append(_NbestPrediction(
            text="empty",
            start_logit=0.0,
            end_logit=0.0,
            start_prob=0.0,
            end_prob=0.0,
            start_word_index=0,
            start_tok_index=0,
        ))
    return nbest


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def qa_evaluate(examples, predictions, lang="en"):
    em = skip = 0
    f1 = precision = recall = 0.0
    for example in examples:
        qas_id = example.qas_id
        answers = [ans["text"] for ans in example.answers]
        if qas_id not in predictions.keys():
            skip += 1
            continue
        pred = predictions[qas_id]
        em += calculate_em(answers, pred, lang)
        metrics = calculate_f1(answers, pred, lang)
        f1 += metrics["f1"]
        precision += metrics["precision"]
        recall += metrics["recall"]

    total = len(examples)
    return {
        "EM": 100.0 * em / total,
        "F1": 100.0 * f1 / total,
        "Precision": 100.0 * precision / total,
        "Recall": 100.0 * recall / total,
        "Skip_count":  skip,
        "Total": total,
    }


def normalize_fr(s):
    def remove_articles(text):
        regex = re.compile(r"\b(le|la|les|l'|au|aux|du|des|un|une)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(remove_articles(lower(s))))


def normalize_en(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_char(s):
    def remove_(text):
        text = re.sub(r"['\"《》<>〈〉\(\)‘’]+", " ", text)
        text = re.sub(r"[，。：？！“”；…·、]+", " ", text)
        text = re.sub(r"[「」（）－～『』]+", " ", text)
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def normalize_answer(ans, lang):
    if lang == "fr":
        return normalize_fr(ans)
    elif lang == "ko" or lang == "zh":
        return normalize_char(ans)
    return normalize_en(ans)


def get_tokens(s, lang):
    if not s:
        return []
    return normalize_answer(s, lang).split()


def calculate_em(answers, pred, lang):
    for ans in answers:
        em = int(normalize_answer(ans, lang) == normalize_answer(pred, lang))
        if em == 1:
            return em
    return 0


def calculate_f1(answers, pred, lang):
    def f1(list1, list2):
        common = Counter(list1) & Counter(list2)
        num_same = sum(common.values())
        if num_same == 0:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        precision = 1.0 * num_same / len(list1)
        recall = 1.0 * num_same / len(list2)
        return {"f1": (2 * precision * recall) / (precision + recall),
                "precision": precision,
                "recall": recall,
        }

    scores = []
    for ans in answers:
        prediction_tokens = get_tokens(pred, lang)
        gt_tokens = get_tokens(ans, lang)
        if lang == "ko" or lang == "zh":
            prediction_char = [c for tok in prediction_tokens for c in tok]
            gt_char = [c for tok in gt_tokens for c in tok]
            scores.append(f1(prediction_char, gt_char))
        else:
            scores.append(f1(prediction_tokens, gt_tokens))
    return max(scores, key=lambda x: x["f1"])
