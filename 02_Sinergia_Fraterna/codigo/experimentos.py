"""Monte Carlo: comparación de reglas de selección de banda en escenarios simulados."""
import sys, json, time
import numpy as np
from multiprocessing import Pool
from mdfd_sinergia import (MDFDTree, kurtogram, synergy_map, cyclic_maps, argmax_node,
                           synergy_descent, diagnostic_score)
from simulador import simulate

FS, L, FD = 12000, 6, 107.3

SCENARIOS = {
    "A_base":            dict(fn=3500),
    "B_impulsos_aleat":  dict(fn=3500, rand_impulses=1.5),
    "C_holgura":         dict(fn=3500, looseness=1.0),
    "D_EMI_tonos":       dict(fn=3500, emi=2.0, tones=2.0),
    "E_frontera_diad":   dict(fn=3000),
}


def select_all(T, K, syn, auto, sib, sfd, z_thr=3.0):
    par = lambda k: k[0] < T.L
    sel = {}
    sel["Kurtogram (K)"] = argmax_node(K)
    sel["Sinergia S"] = argmax_node({k: v["S"] for k, v in syn.items()})
    sel["Sinergia Z"] = argmax_node({k: v["Z"] for k, v in syn.items()})
    sel["Descenso sinergia"] = synergy_descent(T, syn, z_thr)[0]
    ok = {k: K[k] for k, v in syn.items() if v["Z"] >= z_thr}
    sel["K validado por Z"] = argmax_node(ok) if ok else sel["Kurtogram (K)"]
    sel["Dirigido auto (SES)"] = argmax_node(auto)
    sel["Coherencia fraterna"] = argmax_node(sib)
    sel["Sinergia dirigida S_fd"] = argmax_node(sfd)
    return sel


def one_run(args):
    scen, snr, seed = args
    rng = np.random.default_rng(seed)
    x = simulate(fs=FS, fd=FD, snr_db=snr, rng=rng, **SCENARIOS[scen])
    T = MDFDTree(x, FS, L)
    K = kurtogram(T)
    syn = synergy_map(T, n_surr=12, rng=rng)
    auto, sib, sfd = cyclic_maps(T, FD)
    D = {k: diagnostic_score(T.z(*k), FS, FD, bw=T.fc_bw(*k)[1]) for k in T.nodes()}
    oracle = max(D, key=lambda k: D[k][0])
    res = {"scen": scen, "snr": snr, "seed": seed, "oracle": [oracle, D[oracle][0]]}
    for name, node in select_all(T, K, syn, auto, sib, sfd).items():
        fc, bw = T.fc_bw(*node)
        res[name] = dict(node=node, fc=fc, bw=bw, hbr=D[node][0], det=bool(D[node][1]))
    return res


if __name__ == "__main__":
    nrep = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    snrs = [float(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [-10.0]
    out = sys.argv[3] if len(sys.argv) > 3 else "resultados_sim.json"
    jobs = [(s, snr, 1000 * i + j) for s in SCENARIOS for i, snr in enumerate(snrs) for j in range(nrep)]
    t0 = time.time()
    with Pool(2) as p:
        results = p.map(one_run, jobs, chunksize=1)
    json.dump(results, open(out, "w"), default=lambda o: o if not isinstance(o, np.generic) else o.item())
    print("done", len(results), "runs in", round(time.time() - t0), "s")
