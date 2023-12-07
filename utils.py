# encoding = "utf-8"
import dataclasses
import json
from typing import List, Dict

from pandas import DataFrame
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn import metrics
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score


@dataclasses.dataclass
class FFLMScoreBestThresholdPredictions:
    all_best_threshold: float = 0.0
    all_best_f1: float = 0.0
    all_test: List[float] = None
    all_test_labels: List[int] = None


def compute_scores_from_file(file_path: str, skip_label: str = None) -> DataFrame:
    delta_1 = []
    delta_2 = []
    delta_3 = []
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            line = json.loads(line.strip())
            s2s, s2s_doc, lm, lm_doc, prefix, s2s_loss, s2s_loss_doc, lm_loss, lm_loss_doc, prefix_loss = score_calculation(
                line)

            if skip_label and skip_label == line["label"]:
                continue

            score_1 = np.mean(np.exp((1 - s2s)) * (lm_loss - s2s_loss))
            score_2 = np.mean(np.exp((1 - s2s)) * (prefix_loss - s2s_loss))
            score_3 = np.mean(np.exp((1 - s2s_doc)) * (lm_loss_doc - s2s_loss_doc))

            delta_1.append(score_1)
            delta_2.append(score_2)
            delta_3.append(score_3)
            labels.append(line["label"])

        fflm_score_deltas = DataFrame({"delta_1": delta_1, "delta_2": delta_2, "delta_3": delta_3, "labels": labels})
        return fflm_score_deltas


def find_best_threshold_labels(fflm_score_deltas_val: DataFrame,
                               fflm_score_delta_test: DataFrame, alpha_range: List[float] = None,
                               beta_range: List[float] = None) -> FFLMScoreBestThresholdPredictions:
    best_threshold_predictions = FFLMScoreBestThresholdPredictions()
    for index, alpha in enumerate(alpha_range):
        for beta in beta_range:
            val_delta = alpha * fflm_score_deltas_val["delta_1"] + beta * fflm_score_deltas_val["delta_3"] + (
                    1 - alpha - beta) * fflm_score_deltas_val["delta_2"]

            best_threshold, best_f1 = choose_best_threshold(fflm_score_deltas_val["labels"], val_delta)
            if best_f1 > best_threshold_predictions.all_best_f1:
                print(alpha, beta)
                best_threshold_predictions.all_best_threshold = best_threshold
                best_threshold_predictions.all_best_f1 = best_f1
                best_threshold_predictions.all_test = alpha * fflm_score_delta_test["delta_1"] + beta * \
                                                      fflm_score_delta_test["delta_3"] + (1 - alpha - beta) * \
                                                      fflm_score_delta_test["delta_2"]
                best_threshold_predictions.all_test_labels = fflm_score_delta_test["labels"]
    return best_threshold_predictions


def get_multi_label_metrics(predictions: List[int], labels: List[int], f1_macro_class_labels: List[str]) -> Dict:
    metrics = ["bacc", "f1_macro", "f1_all", "precision_macro", "recall_macro"]
    results = {}
    for metric in metrics:
        class_labels = None
        if metric == "f1_all":
            class_labels = f1_macro_class_labels
        result = complex_metric(preds=predictions, labels=labels, metric=metric, class_labels=class_labels)
        print(f"Metric: {metric}: {result}")
        results[metric] = result
    return results


def choose_best_threshold(labels, scores):
    '''following https://github.com/tingofurro/summac'''

    best_f1 = 0.0
    best_thresh = 0.0
    thresholds = [np.percentile(scores, p) for p in np.arange(0, 100, 0.2)]
    for thresh in thresholds:
        preds = [1 if score > thresh else 0 for score in scores]
        f1_score = metrics.balanced_accuracy_score(labels, preds)
        if f1_score >= best_f1:
            best_f1 = f1_score
            best_thresh = thresh
    return best_thresh, best_f1


def get_metrics(predicts, labels, full_score, is_correlation=False, is_balanced_acc=False, threshold=None):
    if is_correlation:
        pearson, _ = pearsonr(predicts, labels)
        print("pearson", pearson)
        spearman, _ = spearmanr(predicts, labels)
        print("spearman", spearman)
        kendall, _ = kendalltau(predicts, labels)
        print("kendall", kendall)

    if is_balanced_acc:
        labels = [0 if x != full_score else 1 for x in labels]
        predicts = [1 if x > threshold else 0 for x in predicts]
        # balanced Acc
        b_acc = metrics.balanced_accuracy_score(y_true=labels, y_pred=predicts)
        print("balanced-accuracy", b_acc)


def complex_metric(preds, labels, metric="bacc", class_labels=None):
    match metric:
        case "bacc":
            return balanced_accuracy_score(y_true=labels, y_pred=preds)
        case "f1_macro":
            return f1_score(y_true=labels, y_pred=preds, average="macro")
        case "f1_all":
            f1_scores_array = f1_score(y_true=labels, y_pred=preds, average=None)
            f1_dict = dict(zip(class_labels, f1_scores_array))
            return f1_dict
        case "f1_micro":
            return f1_score(y_true=labels, y_pred=preds, average="micro")
        case "precision_macro":
            return precision_score(y_true=labels, y_pred=preds, average="macro")
        case "recall_macro":
            return recall_score(y_true=labels, y_pred=preds, average="macro")
        case _:
            raise ValueError(f" Unknown metric {metric}")


def score_calculation(content):
    s2s = content["s2s_tok_list"] if type(content["s2s_tok_list"]) == list else [content["s2s_tok_list"]]
    s2s_doc = content["s2s_tok_list_1"] if type(content["s2s_tok_list_1"]) == list else [content["s2s_tok_list_1"]]
    lm = content["lm_tok_list"] if type(content["lm_tok_list"]) == list else [content["lm_tok_list"]]
    lm_doc = content["lm_tok_list_1"] if type(content["lm_tok_list_1"]) == list else [content["lm_tok_list_1"]]
    prefix = content["prefix_tok_list"] if type(content["prefix_tok_list"]) == list else [content["prefix_tok_list"]]

    s2s = np.array([x if x != 0.0 else float(1e-6) for x in s2s])
    s2s_doc = np.array([x if x != 0.0 else float(1e-6) for x in s2s_doc])
    lm = np.array([x if x != 0.0 else float(1e-6) for x in lm])
    lm_doc = np.array([x if x != 0.0 else float(1e-6) for x in lm_doc])
    prefix = np.array([x if x != 0.0 else float(1e-6) for x in prefix])

    s2s_loss = - np.log(s2s)
    s2s_loss_doc = - np.log(s2s_doc)
    lm_loss = -np.log(lm)
    lm_loss_doc = -np.log(lm_doc)
    prefix_loss = -np.log(prefix)

    return s2s, s2s_doc, lm, lm_doc, prefix, s2s_loss, s2s_loss_doc, lm_loss, lm_loss_doc, prefix_loss
