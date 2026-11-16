"""Evaluación en CWRU (12 kHz DE y FE) de las reglas de selección de banda.

Criterio de diagnóstico (igual para todos los métodos): en el SES de la banda elegida,
la relación armónica/fondo en fd debe ser >= 6 dB (ver diagnostic_score).
Categoría de referencia: mejor resultado de Smith & Randall (2015) para ese sensor (M1-M3).
"""
import sys, glob, json, re, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool
from mdfd_sinergia import (MDFDTree, kurtogram, synergy_map, cyclic_maps, diagnostic_score)
from experimentos import select_all

import os
# Carpeta con subcarpetas 12k_DE/, 12k_FE/, Normal/ (estructura del repo XiongMeijing/CWRU-1)
DATA = os.environ.get("CWRU_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "CWRU"))
MULT = {"DE": dict(BPFI=5.415, BPFO=3.585, FTF=0.3983, BSF=2.357),
        "FE": dict(BPFI=4.947, BPFO=3.053, FTF=0.3816, BSF=1.994)}

# Mejor categoría de Smith & Randall (Y/P/N) para el sensor analizado (tablas B2 y B4)
BENCH = {}
for r in [105, 106, 107, 108, 169, 170, 171, 172, 209, 210, 211, 212, 121, 222, 223,
          130, 131, 132, 133, 197, 234, 235, 236, 237, 144, 145, 146, 147, 246, 247, 248, 249,
          156, 158, 159, 160, 258, 259, 260, 261, 48, 49, 50, 51,
          278, 279, 280, 281, 274, 275, 276, 277, 270, 271, 272, 273, 284,
          294, 295, 296, 297, 313, 315, 309, 310, 311, 312, 316, 317, 318]:
    BENCH[r] = "Y"
for r in [185, 186, 187, 188, 198, 199, 283, 286, 287, 288, 289, 291, 293, 298, 299, 300, 301]:
    BENCH[r] = "P"
for r in [118, 119, 120, 224, 225, 200, 56, 57, 58, 59, 282, 285, 290, 292, 302, 305, 306, 307]:
    BENCH[r] = "N"


def load(path, sensor):
    d = sio.loadmat(path)
    key = [k for k in d if k.endswith(f"_{sensor}_time")][0]
    num = int(re.findall(r"X(\d+)_", key)[0])
    rpm_key = [k for k in d if k.endswith("RPM")]
    rpm = float(np.ravel(d[rpm_key[0]])[0]) if rpm_key else 1797.0
    return np.ravel(d[key]).astype(float), num, rpm


def fault_freq(name, sensor, fr):
    c = MULT[sensor]
    if name.startswith("IR"):
        return c["BPFI"] * fr, "IR"
    if name.startswith("OR"):
        return c["BPFO"] * fr, "OR"
    if name.startswith("B"):
        return 2 * c["BSF"] * fr, "B"
    return None, "N"


def run_file(args):
    path, sensor, fs, L, fd_override = args
    name = path.split("/")[-1][:-4]
    x, num, rpm = load(path, sensor)
    if "/Normal/" in path:                      # línea base registrada a 48 kHz (Smith & Randall, Tabla B1)
        from scipy.signal import decimate
        x = decimate(x, 4, ftype="fir")
    fr = rpm / 60
    fd, ftype = fault_freq(name, sensor, fr)
    if fd_override is not None:
        fd, ftype = fd_override * fr, "Normal"
    T = MDFDTree(x, fs, L)
    K = kurtogram(T)
    syn = synergy_map(T, n_surr=12, rng=np.random.default_rng(num))
    auto, sib, sfd = cyclic_maps(T, fd)
    res = dict(file=name, num=num, sensor=sensor, rpm=rpm, fd=fd, type=ftype,
               bench=BENCH.get(num, "?"))
    for meth, node in select_all(T, K, syn, auto, sib, sfd).items():
        fc, bw = T.fc_bw(*node)
        h, det = diagnostic_score(T.z(*node), fs, fd, bw=bw)
        res[meth] = dict(node=list(node), fc=fc, bw=bw, hbr=h, det=bool(det))
    return res


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = sys.argv[2] if len(sys.argv) > 2 else f"res_cwru_L{L}.json"
    jobs = [(p, "DE", 12000, L, None) for p in sorted(glob.glob(DATA + "/12k_DE/*.mat"))]
    jobs += [(p, "FE", 12000, L, None) for p in sorted(glob.glob(DATA + "/12k_FE/*.mat"))]
    # Sanos: falsa alarma evaluada en BPFO, BPFI y 2*BSF del rodamiento DE
    for p in sorted(glob.glob(DATA + "/Normal/*.mat")):
        for mult in (MULT["DE"]["BPFO"], MULT["DE"]["BPFI"], 2 * MULT["DE"]["BSF"]):
            jobs.append((p, "DE", 12000, L, mult))
    t0 = time.time()
    with Pool(2) as pool:
        R = pool.map(run_file, jobs, chunksize=1)
    json.dump(R, open(out, "w"))
    print("done", len(R), "in", round(time.time() - t0), "s")
