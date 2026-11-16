"""Auditoría local de ambos modelos; no modifica pesos ni resultados históricos.

La prueba temporal mantiene el tipo ya predicho sobre el registro completo:
es una prueba condicional de robustez, no otra validación del clasificador.
"""
from pathlib import Path
import hashlib
import inspect
import json
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.io import loadmat

from modelo_ciego import F
from ranking_catalogo import grupo_condicion
from alinear_bandas import geometria, rasgos_alineados
from criterio_usuario import evaluar, ORDEN
from evaluar_no_retroceso import calificar
from entrenar_48k import crear_modelo
from predictor_espectral import predecir as predecir12, MODELO
from predictor_48k import predecir as predecir48, MODELO_48K
from integridad_senales import detectar_filtraciones

AQUI = Path(__file__).parent
SALIDA = AQUI / 'auditoria_modelos_actuales'
DATOS = AQUI.parent / 'data2/CWRU'


def firma(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cargar(r):
    mat = loadmat(DATOS / r.archivo)
    clave = F.clave_canal(mat, r.sensor_falla, int(r.registro))
    claves = [k for k in mat if k.endswith('RPM')]
    rpm = F.rpm_registro(mat, int(r.registro), nominal=float(r.rpm_nominal))
    catalogo = {k: float(v*rpm/60) for k, v in F.MULTIPLICADORES[r.sensor_falla].items()}
    return np.ravel(mat[clave]), catalogo, rpm, bool(claves)


def grado_banda_fija(x, fs, catalogo, q):
    """Verifica la banda/familia elegida antes, sin buscar otra en la segunda mitad."""
    if q.get('nodo') is None:
        return 'ND'
    l, m = q['nodo']
    ancho = fs / 2**(l+1)
    n = len(x)
    f_abs = abs(np.fft.fftfreq(n, 1/fs))
    mascara = (f_abs >= m*ancho) & (
        (f_abs <= (m+1)*ancho) if m == 2**l-1 else (f_abs < (m+1)*ancho))
    h = np.zeros(n)
    h[0] = 1
    h[1:(n+1)//2] = 2
    if n % 2 == 0:
        h[n//2] = 1
    a = abs(np.fft.ifft(np.fft.fft(x-x.mean()) * mascara * h))
    s = abs(np.fft.rfft(a-a.mean()))**2/n**2
    base = 2*catalogo['BSF'] if q['familia'] == '2BSF' else catalogo[q['familia']]
    return evaluar(np.fft.rfftfreq(n, 1/fs), s, base, H=6, p=q['p_prueba'])['grado']


def main():
    SALIDA.mkdir(exist_ok=True)
    hashes = {str(p.relative_to(AQUI)): firma(p) for p in (MODELO, MODELO_48K)}
    idx = F.leer_indice(str(DATOS))
    globales = pd.read_csv(AQUI / 'resultados_modelos_todos/consolidado_133.csv')
    assert len(idx) == len(globales) == 133
    assert idx.archivo.is_unique and globales.archivo.is_unique
    assert set(idx.archivo) == set(globales.archivo)
    evaluados = globales[globales.evaluable.eq(True)]
    assert len(evaluados) in (129, 131)
    inventario = []
    for r in idx.itertuples():
        x, c, rpm, medido = cargar(r)
        inventario.append(dict(archivo=r.archivo, fs=r.fs, sensor=r.sensor_falla,
            real=r.zona, muestras=len(x), segundos=len(x)/r.fs, rpm=rpm,
            rpm_en_mat=medido, grupo=grupo_condicion(r.archivo),
            sha256_archivo=firma(DATOS/r.archivo),
            sha256_senal=hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()))
    inv = pd.DataFrame(inventario)
    inv.to_csv(SALIDA/'inventario.csv', index=False)
    duplicados = {}
    for col in ['sha256_archivo', 'sha256_senal']:
        duplicados[col] = [g.archivo.tolist() for _, g in inv.groupby(col) if len(g) > 1]
    print('Inventario y duplicados comprobados', flush=True)

    solapamientos = []
    excluidas_copia = set()
    for nombre, ruta in [('12k', AQUI/'particion_desarrollo.csv'),
                         ('48k', AQUI/'resultados_modelo_48k/particion.csv')]:
        split = pd.read_csv(ruta)
        dev = set(split.loc[split.particion.eq('desarrollo'), 'archivo'])
        reserva = set(split.loc[split.particion.eq('reserva'), 'archivo'])
        assert dev.isdisjoint(reserva)
        excluidas_copia.update(detectar_filtraciones(
            inv.set_index('archivo').sha256_senal.to_dict(), dev, reserva))
        grupos_dev = {grupo_condicion(a) for a in dev}
        for archivo in sorted(reserva):
            r = inv.set_index('archivo').loc[archivo]
            solapamientos.append(dict(modelo=nombre, archivo=archivo, real=r.real,
                grupo=r.grupo, condicion_presente_en_entrenamiento=r.grupo in grupos_dev))
    pd.DataFrame(solapamientos).to_csv(SALIDA/'solapamiento_reserva.csv', index=False)

    artefactos48 = AQUI/'resultados_modelo_48k'
    meta48 = pd.read_csv(artefactos48/'validacion.csv')
    with np.load(artefactos48/'rasgos_desarrollo.npz', allow_pickle=False) as datos:
        x48 = datos['x']
    errores = []
    for i, r in enumerate(meta48.itertuples()):
        m = idx.set_index('archivo', drop=False).loc[r.archivo]
        x, c, _, _ = cargar(m)
        recalculado, _, _ = rasgos_alineados(x, 48000, c)
        assert np.allclose(recalculado, x48[i], rtol=1e-10, atol=1e-10)
        errores.append(float(abs(recalculado-x48[i]).max()))
    ajuste = crear_modelo().fit(x48, meta48.real)
    escala, lda = ajuste.steps[0][1], ajuste.steps[-1][1]
    with np.load(MODELO_48K, allow_pickle=False) as modelo:
        for clave, valor in [('media', escala.mean_), ('escala', escala.scale_),
                             ('coef', lda.coef_), ('intercepto', lda.intercept_)]:
            assert np.allclose(modelo[clave], valor)
        assert np.array_equal(modelo['clases'], lda.classes_)
    print('48k: 14 rasgos desde señal cruda y pesos reproducidos', flush=True)

    temporal = []
    referencia = pd.read_csv(AQUI/'resultados_criterio_usuario_total/comparacion.csv').set_index('archivo')
    for i, r in enumerate(evaluados[evaluados.real.ne('N')].itertuples()):
        m = idx.set_index('archivo', drop=False).loc[r.archivo]
        x, c, _, _ = cargar(m)
        mitad = len(x)//2
        nodos, L = geometria(r.fs)
        q = calificar(x[:mitad], r.fs, c, r.prediccion, L=L, criterio='usuario', nodos=nodos)
        segunda = grado_banda_fija(x[mitad:], r.fs, c, q)
        # Solo degradar el grado histórico: no acreditar mejoras nuevas por el ensayo.
        robusto = min([r.grado, q['grado'], segunda], key=ORDEN.get)
        temporal.append(dict(archivo=r.archivo, fs=r.fs, origen=r.origen,
            tipo_predicho=r.prediccion, tipo_correcto=r.tipo_correcto,
            paper=referencia.loc[r.archivo, 'paper'] if r.fs == 12000 else None,
            grado_original=r.grado, seleccion_primera_mitad=q['grado'],
            nodo=str(q.get('nodo')), familia=q.get('familia'),
            verificacion_segunda_mitad=segunda, grado_conservador=robusto))
        if (i+1) % 15 == 0:
            print(f'Persistencia temporal: {i+1}/{int(evaluados.real.ne("N").sum())}', flush=True)
            pd.DataFrame(temporal).to_csv(SALIDA/'persistencia_temporal.csv', index=False)
    temporal = pd.DataFrame(temporal)
    assert len(temporal) == int(evaluados.real.ne('N').sum())
    temporal.to_csv(SALIDA/'persistencia_temporal.csv', index=False)
    temporal[~temporal.archivo.isin(excluidas_copia)].to_csv(
        SALIDA/'persistencia_temporal_sin_copias.csv', index=False)

    controles = []
    rng = np.random.default_rng(20261115)
    for fs in (12000, 48000):
        nodos, L = geometria(fs)
        t = np.arange(fs)/fs
        c = {k: float(v*30) for k, v in F.MULTIPLICADORES['DE'].items()}
        for tipo_control in ('blanco', 'coloreado', 'impulso_aislado', 'modulacion_sin_rodamiento'):
            for rep in range(6):
                x = rng.normal(size=fs)
                if tipo_control == 'coloreado':
                    f = np.fft.rfftfreq(fs, 1/fs)
                    x = np.fft.irfft(np.fft.rfft(x)/np.sqrt(1+f/30), n=fs)
                elif tipo_control == 'impulso_aislado':
                    x *= .01
                    inicio = int(rng.integers(fs//5, 4*fs//5))
                    dt = t[inicio:]-t[inicio]
                    x[inicio:] += 30*np.exp(-dt*250)*np.sin(2*np.pi*2500*dt)
                elif tipo_control == 'modulacion_sin_rodamiento':
                    x = .01*x + (1+.4*np.cos(2*np.pi*30*t))*np.cos(2*np.pi*2500*t)
                for tipo in ('B', 'IR', 'OR'):
                    q = calificar(x, fs, c, tipo, L=L, criterio='usuario', nodos=nodos)
                    controles.append(dict(fs=fs, control=tipo_control, repeticion=rep,
                        hipotesis=tipo, grado=q['grado'], nodo=str(q.get('nodo')),
                        familia=q.get('familia')))
        print(f'Controles sintéticos del calificador: {fs} Hz', flush=True)
    # Los normales también se inspeccionan sin la puerta del clasificador N.
    for r in idx[idx.zona.eq('N')].itertuples():
        x, c, _, _ = cargar(r)
        nodos, L = geometria(r.fs)
        for tipo in ('B', 'IR', 'OR'):
            q = calificar(x, r.fs, c, tipo, L=L, criterio='usuario', nodos=nodos)
            controles.append(dict(fs=r.fs, control='normal_real_'+r.archivo,
                repeticion=0, hipotesis=tipo, grado=q['grado'],
                nodo=str(q.get('nodo')), familia=q.get('familia')))
    controles = pd.DataFrame(controles)
    controles.to_csv(SALIDA/'controles_calificador.csv', index=False)

    resumen_temporal = {}
    for fs, sub in temporal.groupby('fs'):
        correcto = sub.tipo_correcto.eq(True)
        resumen_temporal[str(fs)] = dict(total=len(sub),
            grados_originales=sub.grado_original.value_counts().to_dict(),
            grados_conservadores=sub.grado_conservador.value_counts().to_dict(),
            degradaciones=int(sub.grado_original.ne(sub.grado_conservador).sum()))
        if fs == 12000:
            delta = sub.grado_conservador.map(ORDEN)-sub.paper.map(ORDEN)
            resumen_temporal[str(fs)].update(
                mejoran_tipo_correcto=int((correcto & delta.gt(0)).sum()),
                iguales_tipo_correcto=int((correcto & delta.eq(0)).sum()),
                empeoran_tipo_correcto=int((correcto & delta.lt(0)).sum()),
                tipo_incorrecto=int((~correcto).sum()),
                sin_retroceso_tipo_correcto=int((correcto & delta.ge(0)).sum()))
    firmas_iguales = all(firma(AQUI/p) == h for p, h in hashes.items())
    assert firmas_iguales
    hechos = dict(python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__,
        modelos_sha256=hashes, pesos_no_modificados=firmas_iguales,
        total=133, evaluables=len(evaluados), tipos_correctos=int(evaluados.tipo_correcto.sum()),
        duplicados=duplicados, rasgos48_recalculados=14,
        error_maximo_rasgos48=max(errores), pesos48_reproducidos=True,
        clases_por_fs=pd.crosstab(inv.fs, inv.real).to_dict('index'),
        api12=str(inspect.signature(predecir12)), api48=str(inspect.signature(predecir48)),
        temporal=resumen_temporal,
        aviso_temporal='Prueba condicional: tipo tomado del registro completo. No es OOF nuevo ni test independiente.',
        controles=controles.groupby(['control', 'fs', 'grado']).size().rename('cantidad').reset_index().to_dict('records'))
    (SALIDA/'hechos.json').write_text(json.dumps(hechos, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(resumen_temporal, ensure_ascii=True), flush=True)
    print('Auditoría reproducible completada; pesos intactos.', flush=True)


if __name__ == '__main__':
    main()
