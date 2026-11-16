"""CWRU: detección en la frecuencia verdadera vs. frecuencias SEÑUELO (falsa alarma empírica).

Para cada registro se evalúa cada método en fd (verdadera) y en 2 frecuencias señuelo
que no coinciden con ninguna línea cinemática conocida (armónicos de fr, BPFI, BPFO, BSF,
2BSF, FTF y bandas laterales BPFI±k fr). Un método sólo es útil si detecta en fd
mucho más a menudo que en los señuelos.
Incluye la línea base 'Nodo raíz' (= SES de la señal cruda, Método 1 de Smith & Randall).
"""
import sys, glob, json, time
import numpy as np
from multiprocessing import Pool
from mdfd_sinergia import (MDFDTree, kurtogram, synergy_map, cyclic_maps, argmax_node,
                           synergy_descent, diagnostic_score)
from experimento_cwru import DATA, MULT, BENCH, load, fault_freq

L = 6


def decoys(sensor, n=2, tol=0.035):
    c = MULT[sensor]
    lines = [k for k in range(1, 12)]
    for F in (c["BPFI"], c["BPFO"], c["BSF"], 2 * c["BSF"], c["FTF"]):
        lines += [h * F for h in range(1, 4)]
    lines += [c["BPFI"] + s * k for k in (1, 2) for s in (-1, 1)]
    lines += [2 * c["BSF"] + s * c["FTF"] for s in (-1, 1)]
    cand = np.arange(2.3, 7.0, 0.01)
    dist = np.array([min(abs(x - y) / x for y in lines) for x in cand])
    out = []
    for _ in range(n):
        i = int(np.argmax(dist))
        out.append(round(float(cand[i]), 3))
        dist[np.abs(cand - cand[i]) < 0.8] = -1
    return out


def blind_nodes(T, syn, K, z_thr=3.0):
    sel = {"Kurtogram (K)": argmax_node(K),
           "Sinergia Z": argmax_node({k: v["Z"] for k, v in syn.items()}),
           "Descenso sinergia": synergy_descent(T, syn, z_thr)[0],
           "Nodo raíz (SES crudo)": (0, 0)}
    return sel


def run(args):
    path, sensor = args
    name = path.split("/")[-1][:-4]
    x, num, rpm = load(path, sensor)
    fr = rpm / 60
    fd, ftype = fault_freq(name, sensor, fr)
    T = MDFDTree(x, 12000, L)
    K = kurtogram(T)
    syn = synergy_map(T, n_surr=12, rng=np.random.default_rng(num))
    bl = blind_nodes(T, syn, K)
    res = dict(file=name, num=num, sensor=sensor, type=ftype, bench=BENCH.get(num, "?"), fd=fd,
               levels={k: v[0] for k, v in bl.items()})
    freqs = {"true": fd}
    for j, c in enumerate(decoys(sensor)):
        freqs[f"decoy{j}"] = c * fr
    for tag, f in freqs.items():
        auto, sib, sfd = cyclic_maps(T, f)
        sel = dict(bl)
        sel["Dirigido auto (SES)"] = argmax_node(auto)
        sel["Sinergia dirigida S_fd"] = argmax_node(sfd)
        res[tag] = {m: bool(diagnostic_score(T.z(*n), 12000, f, bw=T.fc_bw(*n)[1])[1]) for m, n in sel.items()}
    return res


if __name__ == "__main__":
    print("señuelos DE:", decoys("DE"), " FE:", decoys("FE"))
    jobs = [(p, "DE") for p in sorted(glob.glob(DATA + "/12k_DE/*.mat"))]
    jobs += [(p, "FE") for p in sorted(glob.glob(DATA + "/12k_FE/*.mat"))]
    t0 = time.time()
    with Pool(2) as pool:
        R = pool.map(run, jobs, chunksize=1)
    json.dump(R, open("res_cwru_senuelos.json", "w"))
    print("done", len(R), round(time.time() - t0), "s")
