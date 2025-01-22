from sklearn.model_selection import ParameterGrid
from river import drift
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

def evaluate_drift_performance(drift_points, expected_drifts, data_stream=None, tolerance=3):
    """
    Evaluates the performance of detected drift points against expected drift points.

    Parameters:
        drift_points (list): List of detected drift points.
        expected_drifts (list): List of true drift points.
        tolerance (int): Tolerance in samples for a drift to be considered correct.

    Returns:
        dict: Evaluation metrics (precision, recall, F1 score).
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched = set()

    for detected in drift_points:
        if any(abs(detected - true) <= tolerance for true in expected_drifts if true not in matched):
            true_positives += 1
            matched.add(next(true for true in expected_drifts if abs(detected - true) <= tolerance))
        else:
            false_positives += 1

    false_negatives = len(expected_drifts) - len(matched)

    true_detection_rate = true_positives / len(expected_drifts) if len(expected_drifts) > 0 else 0
    false_negatives_rate = false_negatives / len(data_stream) if len(data_stream) > 0 else 0
    false_positives_rate = false_positives / len(expected_drifts) if len(expected_drifts) > 0 else 0

    delay = sum(detected - expected for detected, expected in zip(drift_points, expected_drifts)) if expected_drifts else len(data_stream)
    delay = abs(delay / len(drift_points)) if drift_points else len(data_stream)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / len(expected_drifts) if len(expected_drifts) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {
        'true_detection_rate': true_detection_rate,
        'false_negatives_rate': false_negatives_rate,
        'false_positives_rate': false_positives_rate,
        'delay': delay
    }

def optimize_drift_parameters(detection_function, metric_series, expected_drifts, param_grid):
    """
    Optimizes parameters for a drift detection function using a grid search with robust evaluation.

    Parameters:
        detection_function (function): Drift detection function to optimize.
        metric_series (list): Series of metrics for drift detection.
        param_grid (dict): Dictionary with parameters to optimize.
        expected_drifts (list): List of known drift points for evaluation.

    Returns:
        dict: Best parameters and their corresponding evaluation metrics.
    """

    results = []

    for params in ParameterGrid(param_grid):
        try:
            # Modify this check to log rather than skip invalid combinations
            if 'ws' in params and 'ss' in params and params['ss'] >= params['ws']:
                #print(f"Invalid params: {params}. Skipping as stat_size >= window_size.")
                continue

            drift_points = detection_function(metric_series, **params)
            metrics = evaluate_drift_performance(drift_points, expected_drifts, data_stream=metric_series)
            if len(drift_points) > 0:
                results.append({
                    'params': params,
                    'detected_drifts': drift_points,
                    'expected_drifts': expected_drifts,
                    'true_detection_rate': metrics['true_detection_rate'],
                    'false_negatives_rate': metrics['false_negatives_rate'],
                    'false_positives_rate': metrics['false_positives_rate'],
                    'delay': metrics['delay']
                })

        except Exception as e:
            continue

    return pd.DataFrame(results)