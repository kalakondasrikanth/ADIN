#!/usr/bin/env python3
import numpy as np, os, os.path, sys, argparse
from collections import defaultdict

def compute_scores(label_directory, prediction_directory):
    # Set parameters.
    label_header = 'SepsisLabel'
    prediction_header = 'PredictedLabel'
    probability_header = 'PredictedProbability'

    dt_early = -12
    dt_optimal = -6
    dt_late = 3

    max_u_tp = 1
    min_u_fn = -2
    u_fp = -0.05
    u_tn = 0

    # Find label and prediction files.
    label_files = []
    for filename in os.listdir(label_directory):
        full_filename = os.path.join(label_directory, filename)
        if os.path.isfile(full_filename) and full_filename.endswith('.psv'):
            label_files.append(filename)
    label_files = sorted(label_files)

    prediction_files = []
    for filename in os.listdir(prediction_directory):
        full_filename = os.path.join(prediction_directory, filename)
        if os.path.isfile(full_filename) and full_filename.endswith('.psv'):
            prediction_files.append(filename)
    prediction_files = sorted(prediction_files)

    # Load labels and predictions.
    num_files = len(label_files)
    cohort_labels = []
    cohort_predictions = []
    cohort_probabilities = []

    for k in range(num_files):
        labels = load_column(os.path.join(label_directory, label_files[k]), label_header)
        predictions = load_column(os.path.join(prediction_directory, prediction_files[k]), prediction_header)
        probabilities = load_column(os.path.join(prediction_directory, prediction_files[k]), probability_header)

        num_records = len(labels)

        if 0 < np.sum(predictions) < num_records:
            min_probability_positive = np.min(probabilities[predictions == 1])
            max_probability_negative = np.max(probabilities[predictions == 0])

        # Record labels and predictions.
        cohort_labels.append(labels)
        cohort_predictions.append(predictions)
        cohort_probabilities.append(probabilities)

    # Compute AUC, accuracy, and F-measure.
    labels = np.concatenate(cohort_labels)
    predictions = np.concatenate(cohort_predictions)
    probabilities = np.concatenate(cohort_probabilities)

    auroc, auprc = compute_auc(labels, probabilities)
    accuracy, f_measure = compute_accuracy_f_measure(labels, predictions)

    # Compute utility.
    observed_utilities = np.zeros(num_files)
    best_utilities = np.zeros(num_files)
    worst_utilities = np.zeros(num_files)
    inaction_utilities = np.zeros(num_files)

    for k in range(num_files):
        labels = cohort_labels[k]
        num_records = len(labels)
        observed_predictions = cohort_predictions[k]
        best_predictions = np.zeros(num_records)
        worst_predictions = np.zeros(num_records)
        inaction_predictions = np.zeros(num_records)

        if any(labels):
            t_sepsis = min(i for i, label in enumerate(labels) if label)
            best_predictions[max(0, t_sepsis + dt_early + 1): min(t_sepsis + dt_late + 1, num_records - 1)] = 1
        worst_predictions = 1 - best_predictions

        observed_utilities[k] = compute_prediction_utility(labels, observed_predictions, dt_early, dt_optimal, dt_late,
                                                           max_u_tp, min_u_fn, u_fp, u_tn)
        best_utilities[k] = compute_prediction_utility(labels, best_predictions, dt_early, dt_optimal, dt_late,
                                                       max_u_tp, min_u_fn, u_fp, u_tn)
        worst_utilities[k] = compute_prediction_utility(labels, worst_predictions, dt_early, dt_optimal, dt_late,
                                                        max_u_tp, min_u_fn, u_fp, u_tn)
        inaction_utilities[k] = compute_prediction_utility(labels, inaction_predictions, dt_early, dt_optimal, dt_late,
                                                           max_u_tp, min_u_fn, u_fp, u_tn)

    unnormalized_observed_utility = np.sum(observed_utilities)
    unnormalized_best_utility = np.sum(best_utilities)
    unnormalized_worst_utility = np.sum(worst_utilities)
    unnormalized_inaction_utility = np.sum(inaction_utilities)

    print(unnormalized_observed_utility, unnormalized_best_utility, unnormalized_worst_utility,
          unnormalized_inaction_utility)

    if not (
            unnormalized_worst_utility <= unnormalized_best_utility and unnormalized_inaction_utility <= unnormalized_best_utility):
        raise Exception('Optimal utility must be higher than inaction utility.')

    normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (
            unnormalized_best_utility - unnormalized_inaction_utility)

    return auroc, auprc, accuracy, f_measure, normalized_observed_utility


def normalized_utility_score(targets, predictions):

    dt_early = -12
    dt_optimal = -6
    dt_late = 3

    max_u_tp = 1
    min_u_fn = -2
    u_fp = -0.05
    u_tn = 0

    num_files = len(targets)
    # Compute utility.
    observed_utilities = np.zeros(num_files)
    best_utilities = np.zeros(num_files)
    worst_utilities = np.zeros(num_files)
    inaction_utilities = np.zeros(num_files)

    for k in range(num_files):
        labels = targets[k]
        num_records = len(labels)
        observed_predictions = predictions[k]
        best_predictions = np.zeros(num_records)
        worst_predictions = np.zeros(num_records)
        inaction_predictions = np.zeros(num_records)

        if any(labels):
            t_sepsis = min(i for i, label in enumerate(labels) if label)
            best_predictions[max(0, t_sepsis + dt_early + 1): min(t_sepsis + dt_late + 1, num_records - 1)] = 1
        worst_predictions = 1 - best_predictions

        observed_utilities[k] = compute_prediction_utility(labels, observed_predictions, dt_early, dt_optimal, dt_late,
                                                           max_u_tp, min_u_fn, u_fp, u_tn)
        best_utilities[k] = compute_prediction_utility(labels, best_predictions, dt_early, dt_optimal, dt_late,
                                                       max_u_tp, min_u_fn, u_fp, u_tn)
        worst_utilities[k] = compute_prediction_utility(labels, worst_predictions, dt_early, dt_optimal, dt_late,
                                                        max_u_tp, min_u_fn, u_fp, u_tn)
        inaction_utilities[k] = compute_prediction_utility(labels, inaction_predictions, dt_early, dt_optimal, dt_late,
                                                           max_u_tp, min_u_fn, u_fp, u_tn)

    unnormalized_observed_utility = np.sum(observed_utilities)
    unnormalized_best_utility = np.sum(best_utilities)
    unnormalized_worst_utility = np.sum(worst_utilities)
    unnormalized_inaction_utility = np.sum(inaction_utilities)

    if not (
            unnormalized_worst_utility <= unnormalized_best_utility and unnormalized_inaction_utility <= unnormalized_best_utility):
        raise Exception('Optimal utility must be higher than inaction utility.')

    normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (
            unnormalized_best_utility - unnormalized_inaction_utility)

    # return normalized_observed_utility, 0, 0

    targets_flat = np.concatenate(targets)
    predictions_flat = np.concatenate(predictions)
    accuracy, f_measure = compute_accuracy_f_measure(targets_flat, predictions_flat)
    return normalized_observed_utility, accuracy, f_measure


def load_column(filename, *headers):
    header_to_index = defaultdict(list)
    header_to_column = defaultdict(list)
    with open(filename, 'r') as f:
        for i, l in enumerate(f):
            arrs = l.strip().split('|')
            if i == 0:
                for header in headers:
                        header_to_index[header] = arrs.index(header)
            else:
                for header in headers:
                        header_to_column[header].append(float(arrs[header_to_index[header]]))
    columns = [np.array(header_to_column[header]) for header in headers]

    if len(headers) == 1:
        return columns[0]
    else:
        return columns

def compute_auc(labels, predictions):
    n = len(labels)
    # Find prediction thresholds.
    thresholds = np.unique(predictions)[::-1]
    if thresholds[0] != 1:
        thresholds = np.concatenate((np.array([1]), thresholds))

    if thresholds[-1] != 0:
        thresholds = np.concatenate((thresholds, np.array([0])))
    m = len(thresholds)

    # Populate contingency table across prediction thresholds.
    tp = np.zeros(m)
    fp = np.zeros(m)
    fn = np.zeros(m)
    tn = np.zeros(m)

    # Find indices that sort predicted probabilities from largest to smallest.
    idx = np.argsort(predictions)[::-1]

    i = 0
    for j in range(m):
        # Initialize contingency table for j-th prediction threshold.
        if j == 0:
            tp[j] = 0
            fp[j] = 0
            fn[j] = np.sum(labels == 1)
            tn[j] = np.sum(labels == 0)
        else:
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

        # Update contingency table for i-th largest prediction probability.
        while i < n and predictions[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Summarize contingency table.
    tpr = np.zeros(m)
    tnr = np.zeros(m)
    ppv = np.zeros(m)
    npv = np.zeros(m)

    for j in range(m):
        if tp[j] + fn[j]:
            tpr[j] = tp[j] / (tp[j] + fn[j])
        else:
            tpr[j] = 1
        if fp[j] + tn[j]:
            tnr[j] = tn[j] / (fp[j] + tn[j])
        else:
            tnr[j] = 1
        if tp[j] + fp[j]:
            ppv[j] = tp[j] / (tp[j] + fp[j])
        else:
            ppv[j] = 1
        if fn[j] + tn[j]:
            npv[j] = tn[j] / (fn[j] + tn[j])
        else:
            npv[j] = 1

    # Compute AUROC as the area under a piecewise linear function of TPR /
    # sensitivity (x-axis) and TNR / specificity (y-axis) and AUPRC as the area
    # under a piecewise constant of TPR / recall (x-axis) and PPV / precision
    # (y-axis).
    auroc = 0
    auprc = 0
    for j in range(m - 1):
        auroc += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
        auprc += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    return auroc, auprc

def compute_accuracy_f_measure(labels, predictions):
    n = len(labels)
    # Populate contingency table.
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(n):
        if labels[i] and predictions[i]:
            tp += 1
        elif labels[i] and not predictions[i]:
            fp += 1
        elif not labels[i] and predictions[i]:
            fn += 1
        elif not labels[i] and not predictions[i]:
            tn += 1

    # Summarize contingency table.
    if tp + fp + fn + tn:
        accuracy = float(tp + tn) / float(tp + fp + fn + tn)
    else:
        accuracy = 1.0

    if 2 * tp + fp + fn:
        f_measure = float(2 * tp) / float(2 * tp + fp + fn)
    else:
        f_measure = 1.0

    return accuracy, f_measure


def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2,
                               u_fp=-0.05, u_tn=0):
    n = len(labels)
    # Does the patient eventually have sepsis?
    if any(labels):
        is_septic = True
        t_sepsis = min(i for i, label in enumerate(labels) if label)
    else:
        is_septic = False
        t_sepsis = float('inf')

    # Define slopes and intercept points for affine utility functions of the
    # form u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u)


def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate classifiers for cohort.')
    parser.add_argument('-l', '--labels_directory', type=str, required=True, help='Labels directory')
    parser.add_argument('-p', '--predictions_directory', type=str, required=True, help='Predictions directory')
    parser.add_argument('-o', '--output_file', type=str, required=False, help='Output filename')
    return parser


def run(args):
    auroc, auprc, accuracy, f_measure, utility = compute_scores(args.labels_directory, args.predictions_directory)

    output_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility\n{}|{}|{}|{}|{}'.format(auroc, auprc, accuracy, f_measure,
                                                                                    utility)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(output_string)
    else:
        print(output_string)


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
