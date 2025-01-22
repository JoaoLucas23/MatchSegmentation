from river import drift

def detect_kswin_drift(metric_series, a=0.01,ws=10,ss=3,seed=42):
    """
    Detecta pontos de mudança em uma série temporal de métricas.
    """

    # Inicializa o detector KSWIN
    kswin = drift.KSWIN(alpha=a, window_size=ws, stat_size=ss, seed=seed)

    # Lista para armazenar os pontos de detecção de drift
    drift_points = []

    # Processa a série temporal de métricas e verifica se há detecção de mudança
    for i, metric in enumerate(metric_series):
        kswin.update(metric)
        if kswin.drift_detected:
            drift_points.append(i)

    return drift_points

def detect_adwin_drift(stream, d=0.002,c=32,mb=5,mwl=5,gp=10):

    #drift_detector  = drift.ADWIN(delta=d, clock=c, max_buckets=mb, min_window_length=wwl,grace_period=gp)

    adwin = drift.ADWIN(delta=d,clock=c,max_buckets=mb,min_window_length=mwl,grace_period=gp)

    drifts = []

    for i, val in enumerate(stream):
        adwin.update(val)   # Data is processed one sample at a time
        if adwin.drift_detected:
            # The drift detector indicates after each sample if there is a drift in the data
            drifts.append(i)
            #adwin.reset()

    return drifts