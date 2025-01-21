import optuna
from river import drift
import networkx as nx
import numpy as np

from .syntethic_graphs import generate_synthetic_graphs_with_drift

def optimize_hyperparameters(objective, n_trials=100, direction='minimize'):
    """
    Otimiza os hiperparâmetros de um modelo de detecção de mudanças.
    """
    # Criar estudo de otimização
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    return study

# Função para definir o espaço de busca dos hiperparâmetros
def define_search_space(trial):
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
    window_size = trial.suggest_int('window_size', 50, 500)
    stat_size = trial.suggest_int('stat_size', 10, 100)
    return alpha, window_size, stat_size

# Função-objetivo para otimização
def objective(trial):
    # Definir o espaço de busca dos parâmetros
    alpha, window_size, stat_size = define_search_space(trial)
    
    # Inicializar o detector KSWIN com os parâmetros sugeridos
    kswin = drift.KSWIN(alpha=alpha, window_size=window_size, stat_size=stat_size)

    num_graphs = 50
    num_nodes = 12
    n_drifts = 4
    probs_before_drift = [0.05, 0.15, 0.05, 0.2]
    probs_after_drift = [0.15, 0.05, 0.2, 0.25]
    drift_points = [np.random.randint(5, 35) for _ in range(n_drifts)]
    
    metrics = generate_synthetic_graphs_with_drift(num_graphs, num_nodes, probs_before_drift, probs_after_drift, drift_points)
    
    # Variáveis para avaliar o desempenho
    detected_drifts = []
    false_positives = 0
    expected_drifts = drift_points
    
    # Aplicar KSWIN nas métricas
    for i, metric in enumerate(metrics):
        kswin.update(metric)
        if kswin.drift_detected:
            detected_drifts.append(i)
    
    # Avaliar desempenho
    for drift in detected_drifts:
        if drift not in expected_drifts:
            false_positives += 1
    
    # Calcular métricas de desempenho
    detection_rate = len(set(detected_drifts) & set(expected_drifts)) / len(expected_drifts)
    false_positive_rate = false_positives / len(metrics)
    
    # Função objetivo a ser minimizada
    return false_positive_rate - detection_rate
