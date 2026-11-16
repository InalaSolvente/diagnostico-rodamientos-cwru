"""Figuras para el informe (simulación + CWRU)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mdfd_sinergia import (MDFDTree, kurtogram, synergy_map, argmax_node, ses,
                           synergy_spectrum, heredity_terms, diagnostic_score)
from simulador import simulate

OUT = "fig/"
import os
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"Kurtogram (K)": "#eb6834", "Sinergia Z": "#2a78d6", "Sinergia S": "#1baf7a",
       "Descenso sinergia": "#eda100", "K validado por Z": "#e87ba4",
       "Dirigido auto (SES)": "#008300", "Sinergia dirigida S_fd": "#4a3aa7",
       "Coherencia fraterna": "#e34948"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fc", "#b9d0f0", "#2a78d6", "#123a70"])
DIV = LinearSegmentedColormap.from_list("div", ["#c2410c", "#f2c1a6", "#efefed", "#b9d0f0", "#1d5fb0"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 200,
})


def tree_image(values: dict, L: int):
    img = np.full((L + 1, 2 ** L), np.nan)
    for (l, m), v in values.items():
        w = 2 ** (L - l)
        img[l, m * w:(m + 1) * w] = v
    return img


def plot_tree_map(ax, values, L, fs, title, cmap=SEQ, norm=None, mark=None, truth=None, cbl=""):
    img = tree_image(values, L)
    im = ax.imshow(img, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
                   extent=[0, fs / 2, L + 0.5, -0.5])
    ax.set_yticks(range(L + 1))
    ax.set_ylabel("nivel ℓ")
    ax.set_xlabel("frecuencia [Hz]")
    ax.set_title(title, loc="left")
    for l in range(L + 1):
        for m in range(1, 2 ** l):
            x = m * fs / 2 ** (l + 1)
            ax.plot([x, x], [l - 0.5, l + 0.5], color="white", lw=0.4)
    if truth is not None:
        ax.axvspan(truth[0], truth[1], color=INK, alpha=0.08, lw=0)
        ax.axvline(0.5 * (truth[0] + truth[1]), color=INK, lw=0.8, ls=(0, (3, 2)))
    if mark is not None:
        l, m = mark
        x0, x1 = m * fs / 2 ** (l + 1), (m + 1) * fs / 2 ** (l + 1)
        ax.add_patch(plt.Rectangle((x0, l - 0.5), x1 - x0, 1, fill=False, ec=INK, lw=1.6))
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbl, color=INK2)
    cb.outline.set_visible(False)
    return im


def fig_maps_sim(scen_kw, snr, seed, fname, title_extra):
    fs, L = 12000, 6
    rng = np.random.default_rng(seed)
    x = simulate(fs=fs, snr_db=snr, rng=rng, **scen_kw)
    T = MDFDTree(x, fs, L)
    K = kurtogram(T)
    syn = synergy_map(T, n_surr=16, rng=rng)
    S = {k: v["S"] for k, v in syn.items()}
    Z = {k: v["Z"] for k, v in syn.items()}
    kk, kz = argmax_node(K), argmax_node(Z)
    fn = scen_kw["fn"]
    truth = (fn - 0.04 * fn, fn + 0.04 * fn)
    fig, axs = plt.subplots(3, 1, figsize=(6.6, 7.2), constrained_layout=True)
    plot_tree_map(axs[0], K, L, fs, f"Kurtogram K (elige ℓ={kk[0]}, m={kk[1]})", mark=kk, truth=truth, cbl="K")
    lim = max(abs(min(S.values())), abs(max(S.values())))
    plot_tree_map(axs[1], S, L - 1, fs, "Sinergia fraterna S (padres ℓ < L)", cmap=DIV,
                  norm=TwoSlopeNorm(0, -lim, lim), truth=truth, cbl="S")
    zl = max(abs(min(Z.values())), abs(max(Z.values())))
    plot_tree_map(axs[2], Z, L - 1, fs, f"Sinergia estandarizada Z (elige ℓ={kz[0]}, m={kz[1]})", cmap=DIV,
                  norm=TwoSlopeNorm(0, -zl, zl), mark=kz, truth=truth, cbl="Z")
    fig.suptitle(title_extra, x=0.01, ha="left", fontsize=9, color=INK2)
    fig.savefig(OUT + fname)
    plt.close(fig)
    hk = diagnostic_score(T.z(*kk), fs, 107.3, bw=T.fc_bw(*kk)[1])
    hz = diagnostic_score(T.z(*kz), fs, 107.3, bw=T.fc_bw(*kz)[1])
    return dict(K=kk, Z=kz, hbrK=hk, hbrZ=hz)


def fig_heredity(fname):
    """Descomposición exacta K_p = heredada + S a lo largo del camino raíz -> resonancia."""
    fs, L = 12000, 6
    rng = np.random.default_rng(7)
    x = simulate(fs=fs, snr_db=-12, fn=3500, rng=rng)
    T = MDFDTree(x, fs, L)
    labels, inh, syn, kp = [], [], [], []
    for l in range(L):
        m = int(3500 // (fs / 2 ** (l + 1)))
        t = heredity_terms(T.z(l + 1, 2 * m), T.z(l + 1, 2 * m + 1))
        fl, fu = m * fs / 2 ** (l + 1), (m + 1) * fs / 2 ** (l + 1)
        labels.append(f"ℓ={l}\n{fl:.0f}–{fu:.0f}")
        inh.append(t["inherited"]); syn.append(t["S"])
        kp.append(t["inherited"] + t["S"])
    xs = np.arange(L)
    fig, ax = plt.subplots(figsize=(6.6, 2.9), constrained_layout=True)
    w = 0.36
    b1 = ax.bar(xs - w / 2 - 0.01, inh, w, color="#9fb8d8", label="heredada  w²ₐKₐ + w²ᵦKᵦ")
    b2 = ax.bar(xs + w / 2 + 0.01, syn, w, color="#2a78d6", label="sinergia S")
    ax.plot(xs, kp, "o", color=INK, ms=5, label="K del padre (= suma)")
    for i in range(L):
        ax.annotate(f"{syn[i]:.3f}", (xs[i] + w / 2, max(syn[i], 0)),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("curtosis")
    ax.set_title("Camino raíz → resonancia (3,5 kHz): K del padre = heredada + sinergia", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.savefig(OUT + fname)
    plt.close(fig)


def fig_mc(resfile, fname):
    R = json.load(open(resfile))
    scen_lab = {"A_base": "A: base", "B_impulsos_aleat": "B: impulsos\naleatorios", "C_holgura": "C: holgura",
                "D_EMI_tonos": "D: EMI +\ntonos", "E_frontera_diad": "E: frontera\ndiádica"}
    scens = list(scen_lab)
    blind = ["Kurtogram (K)", "Sinergia S", "Sinergia Z", "K validado por Z", "Descenso sinergia"]
    targ = ["Dirigido auto (SES)", "Sinergia dirigida S_fd", "Coherencia fraterna"]
    snrs = [-15.0, -12.0]
    fig, axs = plt.subplots(2, 2, figsize=(8.8, 5.4), constrained_layout=True, sharey=True)
    for j, snr in enumerate(snrs):
        for i, (meths, ttl) in enumerate([(blind, "Ciegas"), (targ, "Dirigidas (usan fd)")]):
            ax = axs[i, j]
            w = 0.8 / len(meths)
            for k, m in enumerate(meths):
                vals = [100 * np.mean([r[m]["det"] for r in R if r["scen"] == s and r["snr"] == snr]) for s in scens]
                xs = np.arange(len(scens)) - 0.4 + w * (k + 0.5)
                ax.bar(xs, vals, w * 0.9, color=COL[m], label=m if j == 1 else None)
                for xx, v in zip(xs, vals):
                    ax.text(xx, v + 1.5, f"{v:.0f}", ha="center", fontsize=5.5, color=INK2, rotation=90)
            ax.set_xticks(range(len(scens))); ax.set_xticklabels([scen_lab[s] for s in scens], fontsize=7)
            ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
            ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            ax.set_title(f"{ttl} · SNR {snr:.0f} dB", loc="left")
            if j == 0:
                ax.set_ylabel("% detección de fd")
            else:
                ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(OUT + fname)
    plt.close(fig)


def fig_ses_vs_synergy(fname):
    """Mismo par de hermanos: SES (auto) del padre vs espectro de sinergia (cruzado)."""
    fs, L, fd = 12000, 6, 107.3
    rng = np.random.default_rng(11)
    x = simulate(fs=fs, snr_db=-16, fn=3500, rng=rng)
    T = MDFDTree(x, fs, L)
    l, m = 3, 4
    a, S = ses(T.z(l, m), fs)
    a2, s = synergy_spectrum(T.z(l + 1, 2 * m), T.z(l + 1, 2 * m + 1), fs)
    sel = (a > 5) & (a < 450)
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 3.9), constrained_layout=True, sharex=True)
    axs[0].plot(a[sel], S[sel] / S[sel].max(), color="#eb6834", lw=1.1)
    axs[0].set_title("SES del padre (3000–3750 Hz): el ruido suma un piso positivo", loc="left")
    axs[1].plot(a2[sel], s[sel] / np.abs(s[sel]).max(), color="#2a78d6", lw=1.1)
    axs[1].axhline(0, color=INK2, lw=0.6)
    axs[1].set_title("Espectro de sinergia s(α) de sus hijos: el ruido tiene media cero", loc="left")
    for ax in axs:
        for h in (1, 2, 3, 4):
            ax.axvline(h * fd, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_ylabel("normalizado")
    axs[1].set_xlabel("frecuencia cíclica α [Hz]   (líneas: armónicas de fd = 107,3 Hz)")
    fig.savefig(OUT + fname)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1:] or ["maps", "her", "mc", "ses"]
    if "maps" in what:
        print(fig_maps_sim(dict(fn=3000), -12, 4, "mapas_frontera.png",
                           "Escenario E: resonancia en 3 kHz (frontera diádica en todos los niveles), SNR −12 dB"))
        print(fig_maps_sim(dict(fn=3500), -15, 3, "mapas_base.png",
                           "Escenario A: resonancia en 3,5 kHz + ruido gaussiano, SNR −15 dB"))
    if "her" in what:
        fig_heredity("herencia.png")
    if "mc" in what:
        fig_mc("res_sim.json", "montecarlo.png")
    if "ses" in what:
        fig_ses_vs_synergy("ses_vs_sinergia.png")


def fig_cwru_example(path, sensor, fname, title):
    from experimento_cwru import load, fault_freq
    name = path.split("/")[-1][:-4]
    x, num, rpm = load(path, sensor)
    fd, _ = fault_freq(name, sensor, rpm / 60)
    fs, L = 12000, 6
    T = MDFDTree(x, fs, L)
    K = kurtogram(T)
    syn = synergy_map(T, n_surr=16, rng=np.random.default_rng(num))
    Z = {k: v["Z"] for k, v in syn.items()}
    from mdfd_sinergia import synergy_descent
    kk, kz, kd = argmax_node(K), argmax_node(Z), synergy_descent(T, syn, 3.0)[0]
    fig = plt.figure(figsize=(7.0, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.25, 1.25, 1, 1])
    plot_tree_map(fig.add_subplot(gs[0]), K, L, fs, f"Kurtogram K → ({kk[0]},{kk[1]})", mark=kk, cbl="K")
    zl = max(abs(min(Z.values())), abs(max(Z.values())))
    plot_tree_map(fig.add_subplot(gs[1]), Z, L - 1, fs, f"Sinergia Z → ({kz[0]},{kz[1]});  descenso → ({kd[0]},{kd[1]})",
                  cmap=DIV, norm=TwoSlopeNorm(0, -zl, zl), mark=kz, cbl="Z")
    for i, (node, lab, c) in enumerate([(kk, "Kurtogram", COL["Kurtogram (K)"]),
                                        (kd, "Descenso por sinergia", COL["Descenso sinergia"])]):
        ax = fig.add_subplot(gs[2 + i])
        a, S = ses(T.z(*node), fs)
        sel = (a > 3) & (a < 4.3 * fd)
        ax.plot(a[sel], S[sel] / S[sel].max(), color=c, lw=1.0)
        for h in (1, 2, 3, 4):
            ax.axvline(h * fd, color=INK2, lw=0.6, ls=(0, (2, 2)))
        hb = diagnostic_score(T.z(*node), fs, fd, bw=T.fc_bw(*node)[1])
        fc, bw = T.fc_bw(*node)
        ax.set_title(f"SES de la banda de {lab}: {fc-bw/2:.0f}–{fc+bw/2:.0f} Hz  (HBR en fd = "
                     f"{hb[0]:.1f} dB)", loc="left")
        ax.set_ylabel("normalizado"); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel(f"frecuencia cíclica α [Hz]   (líneas: armónicas de fd = {fd:.1f} Hz)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK2)
    fig.savefig(OUT + fname)
    plt.close(fig)
    return kk, kz, kd


def fig_cwru_summary(resfile, fname):
    R = json.load(open(resfile))
    meths = ["Kurtogram (K)", "Sinergia Z", "Descenso sinergia", "Nodo raíz (SES crudo)",
             "Dirigido auto (SES)", "Sinergia dirigida S_fd"]
    labs = ["Kurtogram (K)", "Sinergia Z", "Descenso sinerg.", "Nodo raíz (crudo)",
            "Dirigido auto", "Sinergia dir. S_fd"]
    tp = [100 * np.mean([r["true"][m] for r in R]) for m in meths]
    fa = [100 * np.mean([r[d][m] for r in R for d in ("decoy0", "decoy1")]) for m in meths]
    fig, ax = plt.subplots(figsize=(7.4, 3.3), constrained_layout=True)
    xs = np.arange(len(meths)); w = 0.38
    ax.bar(xs - w / 2 - 0.01, tp, w, color="#3d3d3a", label="detección en fd verdadera")
    ax.bar(xs + w / 2 + 0.01, fa, w, color="#c9c7bf", label="falsa alarma en frecuencias señuelo")
    for x_, a, b in zip(xs, tp, fa):
        ax.text(x_ - w / 2, a + 1.5, f"{a:.0f}", ha="center", fontsize=7.5, color=INK)
        ax.text(x_ + w / 2, b + 1.5, f"{b:.0f}", ha="center", fontsize=7.5, color=INK2)
    ax.axvline(3.5, color=GRID, lw=1)
    ax.text(1.5, 108, "ciegas (no usan fd para elegir la banda)", ha="center", fontsize=7.5, color=INK2)
    ax.text(4.5, 108, "dirigidas (usan fd)", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels([f"{l}\nJ = {(a-b)/100:.2f}" for l, a, b in zip(labs, tp, fa)], fontsize=7.5)
    ax.set_ylim(0, 115); ax.set_ylabel("% de registros")
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax.set_title("CWRU 12 kHz, 105 registros con falla (DE y FE): detección vs. falsa alarma", loc="left")
    fig.savefig(OUT + fname)
    plt.close(fig)
