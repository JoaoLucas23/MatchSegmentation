from sklearn.model_selection import ParameterGrid
from river import drift
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from .syntethic_graphs import generate_synthetic_data


def evaluate_drift_performance(expected_drifts, detected_drifts, tolerance_interval):
    """
    Evaluate the performance of a drift detection model.

    Parameters:
    - data_stream: List or array of data points.
    - expected_drifts: List of indices where drifts are expected.
    - detected_drifts: List of indices where drifts were detected by the model.
    - tolerance_interval: Integer representing the acceptable range (in indices) before an expected drift for a detection to be considered true positive.

    Returns:
    - A dictionary containing the calculated metrics.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    detection_delays = []

    # Sort the lists to ensure proper comparison
    expected_drifts = sorted(expected_drifts)
    detected_drifts = sorted(detected_drifts)

    # Track which expected drifts have been detected
    detected_flags = [False] * len(expected_drifts)

    for detected in detected_drifts:
        # Check if the detected drift is within the tolerance interval of any expected drift
        match_found = False

        diff = 10000

        for i, expected in enumerate(expected_drifts):
            if(abs(expected - detected) < diff):
                diff = abs(expected - detected)
            if not detected_flags[i] and expected - tolerance_interval <= detected <= expected + tolerance_interval:
                true_positives += 1
                detected_flags[i] = True
                match_found = True
                break
            
        if not match_found:
            false_positives += 1

        detection_delays.append(diff)

    # Any expected drifts not detected within the tolerance interval are false negatives
    false_negatives = detected_flags.count(False)

    # Calculate metrics
    num_expected_drifts = len(expected_drifts)
    num_detected_drifts = len(detected_drifts)

    true_detection_rate = true_positives / num_expected_drifts if num_expected_drifts > 0 else 0
    false_negative_rate = false_negatives / num_expected_drifts if num_expected_drifts > 0 else 0
    false_positive_rate = false_positives / num_detected_drifts if num_detected_drifts > 0 else 0
    average_detection_delay = sum(detection_delays) / len(detected_drifts) if len(detected_drifts) > 0 else float('inf')
    recall = true_detection_rate
    precision = true_positives / num_detected_drifts if num_detected_drifts > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'true_detection_rate': true_detection_rate,
        'false_negative_rate': false_negative_rate,
        'false_positive_rate': false_positive_rate,
        'delay': average_detection_delay,
        'recall': recall,
        'precision': precision,
        'f1_score': f1_score
    }



def optimize_drift_parameters(detection_function, param_grid, repetitions=3):
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

    for rep in range(repetitions):

        metric_series, expected_drift_points = generate_synthetic_data(n_samples=50, n_drifts=rep+1)

        for id, params in enumerate(ParameterGrid(param_grid)):
            try:
                # Modify this check to log rather than skip invalid combinations
                if 'ws' in params and 'ss' in params and params['ss'] >= params['ws']:
                    #print(f"Invalid params: {params}. Skipping as stat_size >= window_size.")
                    continue

                drift_points = detection_function(metric_series, **params)
                metrics = evaluate_drift_performance(expected_drift_points, drift_points, tolerance_interval=2)
                if len(drift_points) > 0:
                    results.append({
                        'id_params': id,
                        'params': params,
                        'detected_drifts': drift_points,
                        'expected_drifts': expected_drift_points,
                        'true_detection_rate': metrics['true_detection_rate'],
                        'false_negative_rate': metrics['false_negative_rate'],
                        'false_positive_rate': metrics['false_positive_rate'],
                        'delay': metrics['delay'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1_score']
                    })

            except Exception as e:
                continue

    return pd.DataFrame(results)