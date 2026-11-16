"""Características desde señal y catálogo completo; sin frecuencia objetivo ni etiquetas."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from modelo_ciego import validar_catalogo

FAMILIAS = ("BPFI", "BPFO", "BSF", "2BSF", "FTF")
VERSION_DATOS = 'selector_explicito_cwru_v1'  # Invalida pesos/cachés previos al selector de registro correcto.


def evidencia_armonica(f, s, frecuencias):
    """Conserva contrastes de seis armónicas, sin cortar por significancia."""
    eps = max(float(s.max())*1e-15, np.finfo(float).tiny)
    valores = []
    for frecuencia in frecuencias:
        for h in range(1, 7):
            objetivo = h*frecuencia
            delta = max(.02*objetivo, f[1]-f[0])
            lo, hi = np.searchsorted(f, [objetivo-delta, objetivo+delta], side="left")
            a, b = np.searchsorted(f, [max(0, objetivo-.3*frecuencia), objetivo+.3*frecuencia])
            if hi <= lo or lo >= len(s):
                valores.extend([0., 0.])
                continue
            pico = float(s[lo:hi].max())
            fondo = np.r_[s[a:lo], s[hi:b]]
            media = float(fondo.mean()) if len(fondo) else float(s.mean())
            contraste = np.clip(10*np.log10((pico+eps)/(media+eps)), -60, 60)
            fraccion = np.clip(10*np.log10((pico+eps)/(float(s[1:].sum())+eps)), -150, 0)
            valores.extend([float(contraste), float(fraccion)])
    return np.array(valores)


def extraer(x, fs, frecuencias_fabricante, L=4):
    """Devuelve bancos fijos de rasgos, independientes del defecto y del nombre de archivo."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 1024 or not np.isfinite(x).all():
        raise ValueError("Señal inválida")
    if not np.isfinite(fs) or fs <= 0 or not isinstance(L, int) or not 0 <= L <= 6:
        raise ValueError("Muestreo o profundidad inválidos")
    validar_catalogo(frecuencias_fabricante, fs)
    x = x-x.mean()
    rms = np.sqrt(np.mean(x*x))
    x = x/max(rms, np.finfo(float).tiny)
    n = len(x)
    X = np.fft.fft(x)
    f_abs = np.abs(np.fft.fftfreq(n, 1/fs))
    f = np.fft.rfftfreq(n, 1/fs)
    hilbert = np.zeros(n)
    hilbert[0] = 1
    hilbert[1:(n+1)//2] = 2
    if n % 2 == 0:
        hilbert[n//2] = 1
    catalogo = [frecuencias_fabricante[k] if k != "2BSF" else 2*frecuencias_fabricante["BSF"] for k in FAMILIAS]
    potencia = np.abs(np.fft.rfft(x))**2
    psd = np.array([a.sum() for a in np.array_split(potencia[1:], 64)])
    psd = np.log10(psd/max(psd.sum(), np.finfo(float).tiny)+1e-15)
    estadisticas = [np.log10(max(rms,1e-15)), float(np.mean(x**3)), float(np.mean(x**4)), float(np.max(abs(x)))]
    rasgos, ks, hs, estad_bandas = [], [], [], []
    for l in range(L+1):
        ancho = fs/2**(l+1)
        for m in range(2**l):
            mask = (f_abs >= m*ancho) & ((f_abs <= (m+1)*ancho) if m == 2**l-1 else (f_abs < (m+1)*ancho))
            z = np.fft.ifft(X*mask*hilbert)
            e = abs(z-z.mean())**2
            energia = float(e.mean())
            k = float(np.mean(e**2)/energia**2-2) if energia > 1e-20 else 0.
            envolvente = abs(z)
            s = abs(np.fft.rfft(envolvente-envolvente.mean()))**2/n**2
            p = s[1:]/max(float(s[1:].sum()), np.finfo(float).tiny)
            positivos = p[p>0]
            entropia = -float(np.sum(positivos*np.log(positivos)))/np.log(max(len(p),2)) if len(positivos) else 1.
            rasgos.append(evidencia_armonica(f,s,catalogo))
            ks.append(k); hs.append(entropia)
            estad_bandas.extend([np.log10(energia+1e-15), np.log1p(max(k,0)), entropia])
    rasgos = np.asarray(rasgos)
    candidatos = np.argsort(hs, kind="stable")[:int(np.ceil(.25*len(hs)))]
    elegidos = list(dict.fromkeys([int(np.argmax(ks)), int(candidatos[np.argmax(np.asarray(ks)[candidatos])])]))
    agregados = np.r_[rasgos.max(axis=0), np.median(rasgos,axis=0), rasgos.mean(axis=0)]
    doble = np.r_[rasgos[elegidos].max(axis=0), rasgos[elegidos].mean(axis=0)]
    salida = dict(globales=np.r_[estadisticas, psd], cruda=rasgos[0], doble=doble,
                  multibanda=agregados, arbol=np.r_[rasgos.ravel(),estad_bandas])
    salida["combinado"] = np.r_[salida["globales"],salida["cruda"],salida["multibanda"]]
    if not all(np.isfinite(v).all() for v in salida.values()):
        raise ValueError("Características no finitas")
    return salida
