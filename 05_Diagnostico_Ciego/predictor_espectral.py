"""Inferencia supervisada: señal + fs + catálogo completo, sin defecto esperado."""
from pathlib import Path
import hashlib
import numpy as np
from modelo_espectral import extraer

MODELO = Path(__file__).parent/"resultados_espectrales_desarrollo/modelo.npz"


def predecir(x, fs, frecuencias_fabricante, *, modelo=MODELO):
    with np.load(modelo, allow_pickle=False) as m:
        firma = hashlib.sha256((Path(__file__).parent/"modelo_espectral.py").read_bytes()).hexdigest()
        if str(m["extractor_sha256"]) != firma:
            raise ValueError("El extractor cambió: regenere el modelo antes de usarlo")
        rasgos = extraer(x, fs, frecuencias_fabricante, L=int(m["L"]))[str(m["banco"])]
        if rasgos.shape != m["media"].shape:
            raise ValueError("Dimensión de rasgos incompatible")
        scores = ((rasgos-m["media"])/m["escala"])@m["coef"].T+m["intercepto"]
        clases = m["clases"].tolist()
        return dict(tipo=clases[int(np.argmax(scores))],
                    scores={str(c):float(s) for c,s in zip(clases,scores)},
                    nota="Scores discriminantes, no probabilidades calibradas")
