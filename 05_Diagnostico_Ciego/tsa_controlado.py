"""Resta media síncrona del eje; protege TODAS las familias del catálogo."""
import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import resample_poly, resample
from modelo_ciego import validar_catalogo, F
from rasgos_fisicos import giro_catalogo

FS = 12000


def aplicar(x, fs, catalogo):
    x = np.asarray(x, dtype=float)
    if fs not in (12000, 48000) or x.ndim != 1 or len(x) < 5*fs or not np.isfinite(x).all():
        raise ValueError('Se requiere señal finita de al menos 5 s, fs12/48 kHz')
    validar_catalogo(catalogo, FS)
    mitad = len(x)//2
    # Nunca remuestrea a través de la frontera entre selección y verificación.
    a, b = x[:mitad], x[mitad:]
    if fs == 48000:
        a, b = resample_poly(a, 1, 4), resample_poly(b, 1, 4)
    giro = giro_catalogo(catalogo)
    bins = 2*int(np.ceil((FS/2)/giro))
    if len(a) < 20*bins:
        raise ValueError('Pocas muestras para estimar la media síncrona')
    fase = (np.arange(len(a))*giro/FS) % 1
    posicion = fase*bins
    izq = np.floor(posicion).astype(int) % bins
    peso = posicion-np.floor(posicion)
    der = (izq+1) % bins
    sumas = np.bincount(izq, weights=a*(1-peso), minlength=bins)
    sumas += np.bincount(der, weights=a*peso, minlength=bins)
    pesos = np.bincount(izq, weights=1-peso, minlength=bins)
    pesos += np.bincount(der, weights=peso, minlength=bins)
    if np.any(pesos <= 0):
        raise ValueError('Fases sin cobertura; no inventar plantilla')
    plantilla = sumas/pesos
    coef = rfft(plantilla)
    ordenes = np.arange(len(coef))
    frecuencias = ordenes*giro
    bases = np.array([catalogo['BPFI'], catalogo['BPFO'], catalogo['BSF'],
                      2*catalogo['BSF'], catalogo['FTF']])
    protegidas = (bases[:, None]*np.arange(1, 7)).ravel()
    # Protección uniforme, sin conocer qué familia es la verdadera.
    tolerancia = .02*protegidas + FS/len(a)
    proteger = np.any(abs(frecuencias[:, None]-protegidas) <= tolerancia, axis=1)
    permitido = (ordenes > 0) & (frecuencias <= FS/2) & ~proteger
    coef[~permitido] = 0
    plantilla = irfft(coef, n=bins)
    fina = resample(plantilla, 4*bins)
    eje = np.arange(len(fina)+1)/len(fina)
    periodica = np.r_[fina, fina[0]]

    def fondo(t):
        return np.interp((t*giro) % 1, eje, periodica)

    fa = fondo(np.arange(len(a))/FS)
    # Conserva el origen temporal nativo, incluso cuando mitad no es múltiplo de 4.
    fb = fondo(mitad/fs + np.arange(len(b))/FS)
    meta = dict(fs_nativo=int(fs), fs_procesado=FS, giro=float(giro), bins=bins,
        muestras_seleccion_nativas=mitad, muestras_seleccion_procesadas=len(a),
        inicio_verificacion_s=float(mitad/fs), ordenes_retirados=ordenes[permitido].tolist(),
        ordenes_protegidos=ordenes[proteger].tolist(), plantilla=plantilla.tolist(),
        rms_fondo_seleccion=float(np.sqrt(np.mean(fa*fa))),
        ajuste='solo primera mitad; misma plantilla para ambas mitades')
    return a-fa, b-fb, meta


def comprobar():
    rng = np.random.default_rng(512)
    for sensor in ('DE', 'FE'):
        c = {k: 30*v for k, v in F.MULTIPLICADORES[sensor].items()}
        for fs in (12000, 48000):
            t = np.arange(8*fs+1)/fs
            f = 2*c['BSF']
            x = 4*np.cos(2*np.pi*30*t) + np.cos(2*np.pi*f*t) + .001*rng.normal(size=len(t))
            a, b, m = aplicar(x, fs, c)
            tv = len(x)//2/fs + np.arange(len(b))/FS
            amp_giro = 2*abs(np.mean(b*np.exp(-2j*np.pi*30*tv)))
            amp_falla = 2*abs(np.mean(b*np.exp(-2j*np.pi*f*tv)))
            assert amp_giro < .05 and .95 < amp_falla < 1.05, (sensor, fs, amp_giro, amp_falla)
            if sensor == 'FE':
                assert 4 in m['ordenes_protegidos']
            otro = x.copy()
            otro[len(x)//2:] = rng.normal(scale=50, size=len(x)-len(x)//2)
            aa, _, mm = aplicar(otro, fs, c)
            assert np.array_equal(a, aa) and m == mm
            aaa, bbb, _ = aplicar(3*x, fs, c)
            assert np.allclose(aaa, 3*a, atol=1e-10) and np.allclose(bbb, 3*b, atol=1e-10)
    print('OK: quita giro, conserva 2BSF incluso junto a 4x en FE, ajuste sin futuro y escala 12/48 kHz.', flush=True)


if __name__ == '__main__':
    comprobar()
