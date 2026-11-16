import numpy as np
from ranking_catalogo import grupo_condicion, caracteristicas, validar_grupos, ordenar, TIPOS


if __name__ == "__main__":
    assert grupo_condicion("12k_DE/IR007_0.mat") == grupo_condicion("12k_FE/IR007_3.mat")
    assert grupo_condicion("Normal/Normal_0.mat") == grupo_condicion("Normal/Normal_2.mat")
    e = dict(tipo="IR", familia="BPFI", frecuencia_hz=100., nodo=[1,0], armonicas=[1,2],
             score=20., categoria="D", control=False)
    r = dict(evidencia=[e], bandas=[[1,0],[1,1]], archivo="IR007_0.mat", real="IR")
    a, activo = caracteristicas(r)
    r.update(archivo="otro.mat", real="B", fd=1000)
    assert np.array_equal(a, caracteristicas(r)[0])
    assert a[0,4] == .5 and a[0,6] == 1
    assert activo.tolist() == [True, False, False, False]
    assert ordenar([.5,.45,0,0], [True,True,False,False], .1)[0] == "ambiguo"
    assert ordenar([.5,.2,0,0], [True,True,False,False], .1)[0] == "IR"
    assert ordenar(np.zeros(4), np.zeros(4,dtype=bool))[0] == "sin_evidencia"
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6,len(TIPOS),8))
    y = np.zeros((6,len(TIPOS))); y[:,0] = 1
    grupos = ["a","a","b","b","c","c"]
    activos = np.ones_like(y,dtype=bool)
    p, audit = validar_grupos(x,y,grupos,activos)
    y[:2] = 1-y[:2]
    q, _ = validar_grupos(x,y,grupos,activos)
    assert np.allclose(p[:2],q[:2]), "Las etiquetas del grupo excluido afectaron su predicción"
    assert all(r['grupo_excluido'] not in r['grupos_entrenamiento'] for r in audit)
    print("OK: agrupación, características sin etiqueta, abstención y exclusión del grupo de prueba")
