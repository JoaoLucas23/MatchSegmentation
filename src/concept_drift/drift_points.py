from river import drift
import networkx as nx
import numpy as np

def detect_kswin_drift(metric_series, a=0.35,ws=5,ss=2):
    """
    Detecta pontos de mudança em uma série temporal de métricas.
    """

    # Inicializa o detector KSWIN
    kswin = drift.KSWIN(alpha=a, window_size=ws, stat_size=ss)

    # Lista para armazenar os pontos de detecção de drift
    drift_points = []

    # Processa a série temporal de métricas e verifica se há detecção de mudança
    for i, metric in enumerate(metric_series):
        kswin.update(metric)
        if kswin.drift_detected:
            drift_points.append(i)

    return drift_points
