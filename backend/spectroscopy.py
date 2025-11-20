import io
from typing import List, Tuple
import pandas as pd
import numpy as np
from .models import SpectrumFeature


def parse_spectrum_csv(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse CSV bytes into energy and intensity arrays.
    Expected columns: energy, intensity
    """
    buf = io.BytesIO(file_bytes)
    df = pd.read_csv(buf)

    if "energy" not in df.columns or "intensity" not in df.columns:
        raise ValueError("CSV must contain 'energy' and 'intensity' columns")

    energy = df["energy"].to_numpy(dtype=float)
    intensity = df["intensity"].to_numpy(dtype=float)
    return energy, intensity


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    max_val = np.max(intensity)
    if max_val <= 0:
        return intensity
    return intensity / max_val


def detect_peaks(
    energy: np.ndarray,
    intensity: np.ndarray,
    window: int = 5,
    threshold: float = 0.2,
) -> List[SpectrumFeature]:
    """
    Very simple peak detector:
    - A point is a peak if it's a local maximum in a ±window region
      and above 'threshold' in normalized intensity.
    """
    peaks: List[SpectrumFeature] = []
    n = len(intensity)
    if n == 0:
        return peaks

    # Assume intensity is already normalized [0, 1]
    for i in range(window, n - window):
        local = intensity[i - window : i + window + 1]
        if intensity[i] == np.max(local) and intensity[i] >= threshold:
            peaks.append(
                SpectrumFeature(
                    peak_energy=float(energy[i]),
                    peak_intensity=float(intensity[i]),
                )
            )
    return peaks
