import numpy as np

def va_to_bin(valence, arousal, bins=5):
    """
    Maps continuous VA values to discrete bins.
    Valence, Arousal assumed in original scale [1, 5]
    """
    v_bin = int(np.clip((valence - 1) / (4 / bins), 0, bins - 1))
    a_bin = int(np.clip((arousal - 1) / (4 / bins), 0, bins - 1))
    return v_bin, a_bin