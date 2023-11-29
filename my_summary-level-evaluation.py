import dataclasses
import json
from typing import List, Union, Tuple, Optional, Dict

import sklearn
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from utils import *
import argparse


@dataclasses.dataclass
class FFLMScoreDeltas:
    delta_1: List[float]
    delta_2: List[float]
    delta_3: List[float]
    labels: List[Union[int, str]]


@dataclasses.dataclass
class FFLMScoreBestThresholdPredictions:
    all_best_threshold: float = 0.0
    all_best_f1: float = 0.0
    all_test: List[float] = None
    all_test_labels: List[int] = None


def compute_scores_from_file(file_path: str) -> FFLMScoreDeltas:
    delta_1 = []
    delta_2 = []
    delta_3 = []
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            line = json.loads(line.strip())
            s2s, s2s_doc, lm, lm_doc, prefix, s2s_loss, s2s_loss_doc, lm_loss, lm_loss_doc, prefix_loss = score_calculation(
                line)

            score_1 = np.mean(np.exp((1 - s2s)) * (lm_loss - s2s_loss))
            score_2 = np.mean(np.exp((1 - s2s)) * (prefix_loss - s2s_loss))
            score_3 = np.mean(np.exp((1 - s2s_doc)) * (lm_loss_doc - s2s_loss_doc))

            delta_1.append(score_1)
            delta_2.append(score_2)
            delta_3.append(score_3)
            labels.append(line["label"])

        return FFLMScoreDeltas(delta_1=delta_1, delta_2=delta_2, delta_3=delta_3, labels=labels)


# def get_threshold_labels(alpha:float, beta: float, fflm_score_delta_test: FFLMScoreDeltas) -> FFLMScoreBestThresholdPredictions:
#     best_threshold_predictions = FFLMScoreBestThresholdPredictions()
#
#     best_threshold_predictions.all_best_threshold = best_threshold
#     best_threshold_predictions.all_best_f1 = best_f1
#     best_threshold_predictions.all_test = alpha * np.array(fflm_score_delta_test.delta_1) + beta * np.array(
#         fflm_score_delta_test.delta_3) + (1 - alpha - beta) * np.array(fflm_score_delta_test.delta_2)
#     best_threshold_predictions.all_test_labels = fflm_score_delta_test.labels
#     return best_threshold_predictions

def find_best_threshold_labels(fflm_score_deltas_val: FFLMScoreDeltas,
                               fflm_score_delta_test: FFLMScoreDeltas, alpha_range: List[float] = None,
                               beta_range: List[float] = None) -> FFLMScoreBestThresholdPredictions:
    best_threshold_predictions = FFLMScoreBestThresholdPredictions()
    if not alpha_range:
        alpha_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if not beta_range:
        beta_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for index, alpha in enumerate(alpha_range):
        if index == 0:
            index = None
        else:
            index = -index
        adapted_beta_range = beta_range #[:index]  # beta_range[:int((1 - alpha) * 10) + len(alpha_range)-11]
        for beta in adapted_beta_range:
            val_delta = alpha * np.array(fflm_score_deltas_val.delta_1) + beta * np.array(
                fflm_score_deltas_val.delta_3) + (1 - alpha - beta) * np.array(
                fflm_score_deltas_val.delta_2)

            best_threshold, best_f1 = choose_best_threshold(fflm_score_deltas_val.labels, val_delta)
            if best_f1 > best_threshold_predictions.all_best_f1:
                print(alpha, beta)
                best_threshold_predictions.all_best_threshold = best_threshold
                best_threshold_predictions.all_best_f1 = best_f1
                best_threshold_predictions.all_test = alpha * np.array(fflm_score_delta_test.delta_1) + beta * np.array(
                    fflm_score_delta_test.delta_3) + (1 - alpha - beta) * np.array(fflm_score_delta_test.delta_2)
                best_threshold_predictions.all_test_labels = fflm_score_delta_test.labels
    return best_threshold_predictions


def get_modified_labels(labels: List[str], false_label: str) -> List[int]:
    modified_labels = [0 if label == false_label else 1 for label in labels]
    return modified_labels


def get_intrinsic_and_extrinsic_fflm_score_delta(file_path: str, get_unmodified_labels: bool = False) -> Tuple[
    FFLMScoreDeltas, FFLMScoreDeltas, Optional[List[str]]]:
    fflm_score_deltas = compute_scores_from_file(file_path)
    intrinsic_labels = get_modified_labels(labels=fflm_score_deltas.labels,
                                           false_label="Intrinsic Hallucination")
    extrinsic_labels = get_modified_labels(labels=fflm_score_deltas.labels,
                                           false_label="Extrinsic Hallucination")
    fflm_score_deltas_intrinsic = dataclasses.replace(fflm_score_deltas, labels=intrinsic_labels)
    fflm_score_deltas_extrinsic = dataclasses.replace(fflm_score_deltas, labels=extrinsic_labels)

    if get_unmodified_labels:
        return fflm_score_deltas_intrinsic, fflm_score_deltas_extrinsic, fflm_score_deltas.labels

    return fflm_score_deltas_intrinsic, fflm_score_deltas_extrinsic, None


def get_predictions(best_threshold_labels: FFLMScoreBestThresholdPredictions) -> List[int]:
    predictions = [1 if x > best_threshold_labels.all_best_threshold else 0 for x in
                   best_threshold_labels.all_test]

    return predictions


def get_combined_multi_label_predictions_and_labels(best_intrinsic_threshold_labels: FFLMScoreBestThresholdPredictions,
                                                    best_extrinsic_threshold_labels: FFLMScoreBestThresholdPredictions) \
        -> List[int]:
    predicts_intrinsic = get_predictions(best_threshold_labels=best_intrinsic_threshold_labels)
    predicts_extrinsic = get_predictions(best_threshold_labels=best_extrinsic_threshold_labels)

    # Find Intersection of predictions where prediction is one
    # 0: Intrinsic, 1: Extrinsic, 2: Faithful
    faithful_indices = [i for i, (a, b) in enumerate(zip(predicts_intrinsic, predicts_extrinsic)) if a == 1 and b == 1]
    combined_predictions = predicts_intrinsic
    for faithful_index in faithful_indices:
        combined_predictions[faithful_index] = 2
    return combined_predictions


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


def get_multi_label_metrics(predictions: List[int], labels: List[int]) -> Dict:
    metrics = ["bacc", "f1_macro", "f1_all", "precision_macro", "recall_macro"]
    results = {}
    for metric in metrics:
        class_labels = None
        if metric == "f1_all":
            class_labels = ["Intrinsic Hallucination", "Extrinsic Hallucination", "Faithful"]
        result = complex_metric(preds=predictions, labels=labels, metric=metric, class_labels=class_labels)
        print(f"Metric: {metric}: {result}")
        results[metric] = result
    return results


def multi_label_faithfulness_detection_eval(file_path_test, file_path_val):
    fflm_score_deltas_intrinsic_val, fflm_score_deltas_extrinsic_val, _ = get_intrinsic_and_extrinsic_fflm_score_delta(
        file_path=file_path_val)

    fflm_score_deltas_intrinsic_test, fflm_score_deltas_extrinsic_test, unmodified_labels = get_intrinsic_and_extrinsic_fflm_score_delta(
        file_path=file_path_test, get_unmodified_labels=True)
    combined_labels = []
    for label in unmodified_labels:
        if label == "Intrinsic Hallucination":
            combined_labels.append(0)
        elif label == "Extrinsic Hallucination":
            combined_labels.append(1)
        else:
            combined_labels.append(2)

    best_alpha_intrinsic = 0.25
    best_beta_intrinsic = 0.75
    best_alpha_extrinsic = 0.35
    best_beta_extrinsic = 0.65
    start_range = -0.5
    end_range = 1.0
    step_size = 0.25
    values_range = [
        best_alpha_intrinsic]  # np.round(np.arange(start_range, end_range + step_size, step_size), 2).tolist()
    test_range = np.round(np.arange(start_range, end_range + step_size, step_size), 2).tolist()
    best_results = {"f1_macro": -1}
    for best_alpha_intrinsic in [1.0]:#test_range:
        for best_beta_intrinsic in [0.25]:#test_range:
            for best_alpha_extrinsic in [0.25]:
                for best_beta_extrinsic in [-0.25]:
                    alpha_range_intrinsic = [best_alpha_intrinsic]  # values_range
                    beta_range_intrinsic = [best_beta_intrinsic]
                    alpha_range_extrinsic = [best_alpha_extrinsic]
                    beta_range_extrinsic = [best_beta_extrinsic]
                    print("Intrinsic Threshold")
                    best_intrinsic_threshold_labels = find_best_threshold_labels(
                        fflm_score_deltas_val=fflm_score_deltas_intrinsic_val,
                        fflm_score_delta_test=fflm_score_deltas_intrinsic_test,
                        alpha_range=alpha_range_intrinsic,
                        beta_range=beta_range_intrinsic)
                    print("Extrinsic Threshold")
                    best_extrinsic_threshold_labels = find_best_threshold_labels(
                        fflm_score_deltas_val=fflm_score_deltas_extrinsic_val,
                        fflm_score_delta_test=fflm_score_deltas_extrinsic_test,
                        alpha_range=alpha_range_extrinsic,
                        beta_range=beta_range_extrinsic)
                    combined_predictions = get_combined_multi_label_predictions_and_labels(
                        best_intrinsic_threshold_labels,
                        best_extrinsic_threshold_labels)
                    results = get_multi_label_metrics(predictions=combined_predictions, labels=combined_labels)
                    if results["f1_macro"] > best_results["f1_macro"]:
                        best_results["bacc"] = results["bacc"]
                        best_results["f1_macro"] = results["f1_macro"]
                        best_results["best_alpha_intrinsic"] = best_alpha_intrinsic
                        best_results["best_beta_intrinsic"] = best_beta_intrinsic
                        best_results["best_alpha_extrinsic"] = best_alpha_extrinsic
                        best_results["best_beta_extrinsic"] = best_beta_extrinsic

                    print(results)
    print("Best results:")
    print(best_results)
                    # get_metrics(all_test, all_test_labels, 1, is_balanced_acc=True, threshold=all_best_threshold)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path_test", type=str,
                        default="")
    parser.add_argument("--file_path_val", type=str,
                        default="")
    args = parser.parse_args()

    if "mtc-" in args.file_path_test:
        multi_label_faithfulness_detection_eval(args.file_path_test, args.file_path_val)
