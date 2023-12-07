from utils import *
import argparse


def get_modified_labels(labels: List[str], true_label: str, neutral_label: str = None) -> List[int]:
    modified_labels = []
    for label in labels:
        if label == true_label:
            modified_labels.append(1)
        else:
            if neutral_label and label == neutral_label:
                modified_labels.append(2)
            else:
                modified_labels.append(0)
    return modified_labels


def get_predictions(best_threshold_labels: FFLMScoreBestThresholdPredictions) -> List[int]:
    predictions = [1 if x > best_threshold_labels.all_best_threshold else 0 for x in
                   best_threshold_labels.all_test]

    return predictions


def get_best_predictions(fflm_score_deltas_val: DataFrame, fflm_score_deltas_test: DataFrame, alpha_range: List[float],
                         beta_range: List[float], class_labels: List[str]):
    best_faithful_threshold_labels = find_best_threshold_labels(
        fflm_score_deltas_val=fflm_score_deltas_val,
        fflm_score_delta_test=fflm_score_deltas_test,
        alpha_range=alpha_range,
        beta_range=beta_range)
    predictions = get_predictions(best_threshold_labels=best_faithful_threshold_labels)
    labels = fflm_score_deltas_test["labels"]
    results = get_multi_label_metrics(predictions=predictions,
                                      labels=labels, f1_macro_class_labels=class_labels)
    print(results)

    return predictions


def get_fflm_score_delta_with_modified_labels(fflm_score_deltas: DataFrame, true_label: str,
                                              neutral_label: str = None) -> DataFrame:
    fflm_score_deltas = fflm_score_deltas.copy()
    modified_labels = get_modified_labels(labels=fflm_score_deltas["labels"],
                                          true_label=true_label,
                                          neutral_label=neutral_label)
    fflm_score_deltas["labels"] = modified_labels

    return fflm_score_deltas


def filter_unfaithful_only_validation_samples(fflm_score_deltas_val: DataFrame) -> DataFrame:
    unfaithful_fflm_score_deltas_val = fflm_score_deltas_val[fflm_score_deltas_val["labels"] != "Faithful"]
    return unfaithful_fflm_score_deltas_val


def filter_unfaithful_predicted_samples(fflm_score_deltas_test: DataFrame,
                                        predictions_faithful_unfaithful: List[int]) -> DataFrame:
    assert len(predictions_faithful_unfaithful) == fflm_score_deltas_test.shape[0]
    indices_to_remove = [index for index, element in enumerate(predictions_faithful_unfaithful) if element == 1]
    unfaithful_only_fflm_score_deltas = fflm_score_deltas_test.drop(indices_to_remove)
    return unfaithful_only_fflm_score_deltas


def hallucination_detection_eval(fflm_score_deltas_val: DataFrame, fflm_score_deltas_test: DataFrame,
                                 predictions_faithful_unfaithful: List[int]):
    unfaithful_fflm_score_deltas_val = filter_unfaithful_only_validation_samples(
        fflm_score_deltas_val=fflm_score_deltas_val)
    fflm_score_deltas_intrinsic_extrinsic_val = get_fflm_score_delta_with_modified_labels(
        fflm_score_deltas=unfaithful_fflm_score_deltas_val, true_label="Extrinsic Hallucination")

    unfaithful_predicted_fflm_score_deltas_test = filter_unfaithful_predicted_samples(
        fflm_score_deltas_test=fflm_score_deltas_test, predictions_faithful_unfaithful=predictions_faithful_unfaithful)
    fflm_score_deltas_intrinsic_extrinsic_test = get_fflm_score_delta_with_modified_labels(
        fflm_score_deltas=unfaithful_predicted_fflm_score_deltas_test, true_label="Extrinsic Hallucination",
        neutral_label="Faithful")

    class_labels = ["Intrinsic", "Extrinsic", "Faithful"]
    start_range = 0.0
    end_range = 1.0
    step_size = 0.25
    values_range = np.round(np.arange(start_range, end_range + step_size, step_size), 2).tolist()
    predictions_intrinsic_extrinsic = get_best_predictions(
        fflm_score_deltas_val=fflm_score_deltas_intrinsic_extrinsic_val,
        fflm_score_deltas_test=fflm_score_deltas_intrinsic_extrinsic_test, alpha_range=values_range,
        beta_range=values_range, class_labels=class_labels)
    return predictions_intrinsic_extrinsic


def faithfulness_detection_eval(fflm_score_deltas_val: DataFrame, fflm_score_deltas_test: DataFrame):
    fflm_score_deltas_faithful_unfaithful_val = get_fflm_score_delta_with_modified_labels(
        fflm_score_deltas=fflm_score_deltas_val, true_label="Faithful")

    fflm_score_deltas_faithful_unfaithful_test = get_fflm_score_delta_with_modified_labels(
        fflm_score_deltas=fflm_score_deltas_test, true_label="Faithful")

    class_labels = ["Unfaithful", "Faithful"]
    start_range = 0.0
    end_range = 1.0
    step_size = 0.25
    values_range = np.round(np.arange(start_range, end_range + step_size, step_size), 2).tolist()
    predictions_faithful_unfaithful = get_best_predictions(
        fflm_score_deltas_val=fflm_score_deltas_faithful_unfaithful_val,
        fflm_score_deltas_test=fflm_score_deltas_faithful_unfaithful_test, alpha_range=values_range,
        beta_range=values_range, class_labels=class_labels)
    return predictions_faithful_unfaithful


def multi_level_classification_evaluation(file_path_val: str, file_path_test: str):
    fflm_score_deltas_val_df = compute_scores_from_file(file_path_val)
    fflm_score_deltas_test_df = compute_scores_from_file(file_path_test)
    predictions_faithful_unfaithful = faithfulness_detection_eval(fflm_score_deltas_val=fflm_score_deltas_val_df,
                                                                  fflm_score_deltas_test=fflm_score_deltas_test_df)
    predictions_intrinsic_extrinsic = hallucination_detection_eval(fflm_score_deltas_val=fflm_score_deltas_val_df,
                                                                   fflm_score_deltas_test=fflm_score_deltas_test_df,
                                                                   predictions_faithful_unfaithful=
                                                                   predictions_faithful_unfaithful)
    final_predictions = predictions_faithful_unfaithful
    index = 0
    for final_index, final_prediction in enumerate(final_predictions):
        if final_prediction == 0:
            final_predictions[final_index] = predictions_intrinsic_extrinsic[index]
            index += 1
        else:
            final_predictions[final_index] = 2

    combined_labels = []
    for label in fflm_score_deltas_test_df["labels"]:
        if label == "Intrinsic Hallucination":
            combined_labels.append(0)
        elif label == "Extrinsic Hallucination":
            combined_labels.append(1)
        else:
            combined_labels.append(2)

    class_labels = ["Intrinsic Hallucination", "Extrinsic Hallucination", "Faithful"]
    results = get_multi_label_metrics(predictions=final_predictions,
                                      labels=combined_labels, f1_macro_class_labels=class_labels)
    print("Final Result:")
    print("----------------------")
    print(results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path_test", type=str,
                        default="output/mtc-final_german_faithfulness_benchmark_with_explanations-fflm-mistral7b_test"
                                ".jsonl")
    parser.add_argument("--file_path_val", type=str,
                        default="output/mtc-final_german_faithfulness_benchmark_with_explanations-fflm"
                                "-mistral7b_train.jsonl")
    args = parser.parse_args()

    if "mtc-" in args.file_path_test:
        multi_level_classification_evaluation(args.file_path_val, args.file_path_test)
