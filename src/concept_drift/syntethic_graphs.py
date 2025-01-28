import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

def generate_synthetic_data(n_samples=50, n_drifts=3, value_range=(0, 5)):
    """
    Generates a synthetic dataset with concept drifts.

    Parameters:
        n_samples (int): Total number of samples in the dataset.
        n_drifts (int): Number of drift points in the dataset.
        value_range (tuple): Range of values for the generated data.

    Returns:
        tuple: A list containing the synthetic dataset and the drift points.
    """
    data = []
    drift_points = []

    remaining_samples = n_samples
    for drift in range(n_drifts + 1):
        # Determine the number of samples for this segment dynamically
        if drift == n_drifts:
            samples_in_segment = remaining_samples
        else:
            samples_in_segment = np.random.randint(1, remaining_samples // (n_drifts - drift + 1))

        remaining_samples -= samples_in_segment

        # Generate the data for the segment
        segment = np.random.uniform(value_range[0], value_range[1], samples_in_segment)

        # Simulate a drift by adding a shift to the values
        if drift > 0:
            shift = np.random.uniform(0, 5)
            segment += shift
            drift_points.append(len(data))

        data.extend(segment)

    return data, drift_points

def draw_synthetic_data(synthetic_data, drift_points, detected_drifts=None):
    # Visualize the drift points
    plt.figure(figsize=(10, 6))
    for drift in drift_points:
        plt.axvline(x=drift, color='red', linestyle='--', label='Drift Point' if drift == drift_points[0] else "")

    for drift in detected_drifts or []:
        plt.axvline(x=drift, color='green', linestyle='--', label='Detected Drift' if drift == detected_drifts[0] else "")

    plt.plot(range(len(synthetic_data)), synthetic_data, label='Feature 0')
    plt.xlabel('Sample Index')
    plt.ylabel('Feature Value')
    plt.title('Synthetic Data with Concept Drift')
    plt.legend()
    plt.show()