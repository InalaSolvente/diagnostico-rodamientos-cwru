"""Selección espectral usando solo el segmento de selección y el tipo predicho.

No recibe etiquetas, referencia del paper, rutas ni un segmento de verificación.
No suprime picos a la izquierda de la fundamental para mejorar el grado.
"""
import numpy as np
from scipy.fft import fft, ifft, rfft, rfftfreq, fftfreq, next_fast_len
from alinear_bandas import geometria
from modelo_ciego import validar_catalogo
from criterio_usuario import evaluar, ORDEN

TRANSFORMACIONES = ('amplitud', 'cuadrada')


def validar_senal(x, fs, catalogo):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 4096 or not np.isfinite(x).all():
        raise ValueError('Se requiere señal finita unidimensional de al menos 4096 muestras')
    validar_catalogo(catalogo, fs)
    return x


def espectro(a, fs):
    """Periodograma Hann; no borra frecuencias ni modifica el umbral del juez."""
    a = a-a.mean()
    w = np.hanning(len(a))
    nfft = next_fast_len(len(a))
    s = abs(rfft(a*w, n=nfft))**2/max(float((w*w).sum()), np.finfo(float).tiny)
    return rfftfreq(nfft, 1/fs), s


def envolvente(x, fs, nodo):
    """Mismo banco diádico físico; extensión reflejada para reducir bordes FFT."""
    pad = min(int(.1*fs), len(x)//10)
    xp = np.pad(x-x.mean(), pad, mode='reflect')
    nfft = next_fast_len(len(xp))
    f = abs(fftfreq(nfft, 1/fs))
    l, m = nodo
    ancho = fs/2**(l+1)
    mask = (f >= m*ancho) & ((f <= (m+1)*ancho) if m == 2**l-1 else (f < (m+1)*ancho))
    h = np.zeros(nfft)
    h[0] = 1
    h[1:(nfft+1)//2] = 2
    if nfft % 2 == 0:
        h[nfft//2] = 1
    z = ifft(fft(xp, n=nfft)*mask*h)
    return abs(z[pad:pad+len(x)])


def transformar(a, nombre):
    if nombre not in TRANSFORMACIONES:
        raise ValueError('Transformación inválida')
    return a if nombre == 'amplitud' else a*a


def margenes(q):
    """Margen D: fundamental, otra armónica y dominancia izquierda simultáneas."""
    eps = np.finfo(float).tiny
    evidencias = {a['h']: a for a in q['armonicas']}
    ratios = {h: a['potencia']/max(a['fondo']*a['umbral'], eps)
              if a['significativa'] else 0. for h, a in evidencias.items()}
    primero = evidencias.get(1, {}).get('potencia', 0.)
    dominio = primero/max(q['max_izquierda'], eps)
    d = min(ratios.get(1, 0.), max([v for h, v in ratios.items() if h != 1] or [0.]), dominio)
    return float(np.log10(max(d, 1e-12))), float(np.log10(max(max(ratios.values(), default=0.), 1e-12)))


def seleccionar(x, fs, catalogo, tipo_predicho):
    """Fija banda/familia/transformación antes de disponer del otro segmento."""
    x = validar_senal(x, fs, catalogo)
    nodos, _ = geometria(fs)
    familias = {'IR': [('BPFI', catalogo['BPFI'])],
                'OR': [('BPFO', catalogo['BPFO'])],
                'B': [('BSF', catalogo['BSF']), ('2BSF', 2*catalogo['BSF'])],
                'FTF': [('FTF', catalogo['FTF'])]}
    if tipo_predicho == 'N':
        return dict(grado='ND', nodo=None, motivo='tipo_normal')
    if tipo_predicho not in familias:
        raise ValueError('Tipo no cubierto')
    # Incluye las dos representaciones; nominal, pendiente de calibración empírica.
    p = .01/(len(nodos)*5*6*len(TRANSFORMACIONES))
    mejor = None
    for nodo in nodos:
        a = envolvente(x, fs, nodo)
        for transformacion in TRANSFORMACIONES:
            b = transformar(a, transformacion)
            f, s = espectro(b, fs)
            segmentos = [espectro(v, fs) for v in np.array_split(b, 3)]
            for familia, base in familias[tipo_predicho]:
                q = evaluar(f, s, base, H=6, p=p)
                partes = [evaluar(fp, sp, base, H=6, p=p) for fp, sp in segmentos]
                md = [margenes(v)[0] for v in partes]
                mp = [margenes(v)[1] for v in partes]
                robusto_d = float(np.quantile(md, .25))
                robusto_p = float(np.quantile(mp, .25))
                # Se comparan TODOS los candidatos: sin retorno anticipado al primer D.
                clave = (ORDEN[q['grado']], robusto_d if q['grado'] == 'D' else robusto_p,
                         len(q['identificadas']), margenes(q)[0])
                if mejor is None or clave > mejor[0]:
                    mejor = (clave, dict(grado=q['grado'], nodo=list(nodo), familia=familia,
                        frecuencia_base=float(base), transformacion=transformacion, p_prueba=p,
                        margen_d_segmentos=md, margen_p_segmentos=mp, evidencia=q))
    return mejor[1]


def verificar(x, fs, catalogo, seleccion):
    """Evalúa únicamente la banda/familia fijada. Nunca busca otra en verificación."""
    x = validar_senal(x, fs, catalogo)
    if seleccion['nodo'] is None:
        return dict(grado='ND', evidencia=None)
    a = transformar(envolvente(x, fs, seleccion['nodo']), seleccion['transformacion'])
    f, s = espectro(a, fs)
    q = evaluar(f, s, seleccion['frecuencia_base'], H=6, p=seleccion['p_prueba'])
    return dict(grado=min([seleccion['grado'], q['grado']], key=ORDEN.get), evidencia=q)


if __name__ == '__main__':
    fs = 12000
    t = np.arange(4*fs)/fs
    rng = np.random.default_rng(851)
    catalogo = dict(BPFI=100., BPFO=137., BSF=83., FTF=17.)
    x = (1+.4*np.cos(2*np.pi*100*t)+.2*np.cos(2*np.pi*200*t))*np.cos(2*np.pi*2500*t)
    x += .005*rng.normal(size=len(t))
    q = seleccionar(x[:2*fs], fs, catalogo, 'IR')
    assert q['grado'] == 'D' and verificar(x[2*fs:], fs, catalogo, q)['grado'] == 'D'
    ruido = rng.normal(size=2*fs)
    assert verificar(ruido, fs, catalogo, q)['grado'] != 'D'
    assert verificar(x[2*fs:], fs, catalogo, {**q, 'grado': 'PD'})['grado'] != 'D'
    assert seleccionar(x[:2*fs], fs, catalogo, 'N')['grado'] == 'ND'
    print('OK: banda fijada en selección; armónicos persistentes D y verificación con ruido no D')
