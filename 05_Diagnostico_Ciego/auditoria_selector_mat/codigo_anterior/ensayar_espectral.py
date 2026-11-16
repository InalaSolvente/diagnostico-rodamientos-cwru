"""Ensayos supervisados fuera de grupo, solo desarrollo y sin fd esperada en rasgos."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from modelo_espectral import extraer
from modelo_ciego import F
from ranking_catalogo import grupo_condicion

AQUI = Path(__file__).parent


def extraer_desarrollo(datos, salida):
    particion = pd.read_csv(AQUI/"particion_desarrollo.csv")
    dev = set(particion.loc[particion.particion == "desarrollo", "archivo"])
    idx = F.leer_indice(str(datos))
    idx = idx[idx.archivo.isin(dev)]
    if set(idx.archivo) != dev:
        raise ValueError("Desarrollo incompleto")
    cache = salida/"cache"
    cache.mkdir(parents=True,exist_ok=True)
    version = hashlib.sha256((AQUI/"modelo_espectral.py").read_bytes()).hexdigest()
    registros, bancos = [], {}
    for _, r in idx.iterrows():
        ruta = datos/r.archivo
        firma = hashlib.sha256(f"{r.archivo}|{ruta.stat().st_size}|{ruta.stat().st_mtime_ns}|{version}".encode()).hexdigest()
        archivo_cache = cache/f"{firma}.npz"
        if archivo_cache.exists():
            with np.load(archivo_cache, allow_pickle=False) as c:
                features = {k:c[k] for k in c.files}
        else:
            mat = loadmat(ruta)
            clave = next(k for k in mat if k.endswith(f"_{r.sensor_falla}_time"))
            rpm_keys = [k for k in mat if k.endswith("RPM")]
            rpm = float(np.ravel(mat[rpm_keys[0]])[0]) if rpm_keys else float(r.rpm_nominal)
            catalogo = {k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
            features = extraer(np.ravel(mat[clave]),float(r.fs),catalogo)
            np.savez_compressed(archivo_cache,**features)
        for k,v in features.items():
            bancos.setdefault(k,[]).append(v)
        registros.append(dict(archivo=r.archivo, grupo=grupo_condicion(r.archivo), real=r.zona))
        print(f"Rasgos {len(registros)}/{len(idx)} {r.archivo}",flush=True)
    return pd.DataFrame(registros),{k:np.asarray(v) for k,v in bancos.items()}


def modelos():
    return {
        "svm_c1":lambda:make_pipeline(StandardScaler(),SVC(C=1,gamma="scale",class_weight="balanced")),
        "svm_c10":lambda:make_pipeline(StandardScaler(),SVC(C=10,gamma="scale",class_weight="balanced")),
        "lda":lambda:make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")),
        "arboles":lambda:ExtraTreesClassifier(n_estimators=160,max_features=.5,class_weight="balanced",random_state=2026,n_jobs=1),
    }


def comparar(meta,bancos,salida):
    y=meta.real.to_numpy(); grupos=meta.grupo.to_numpy()
    resumen=[]; registros=[]; auditoria=[]
    for banco,x in bancos.items():
        for nombre,crear in modelos().items():
            pred=np.empty(len(y),dtype=object)
            for grupo in sorted(set(grupos)):
                test=grupos==grupo; train=~test
                clf=crear()
                clf.fit(x[train],y[train])
                pred[test]=clf.predict(x[test])
                auditoria.append(dict(banco=banco,modelo=nombre,grupo_excluido=grupo,
                                     grupos_train=sorted(set(grupos[train]))))
            fallas=y!="N"; sanos=~fallas
            resumen.append(dict(banco=banco,modelo=nombre,aciertos=int((pred[fallas]==y[fallas]).sum()),
                fallas=int(fallas.sum()),porcentaje=round(100*np.mean(pred[fallas]==y[fallas]),2),
                aciertos_total=int((pred==y).sum()),total=len(y),porcentaje_total=round(100*np.mean(pred==y),2),
                alertas_sanos=int((pred[sanos]!="N").sum()),sanos=int(sanos.sum())))
            registros.extend(dict(archivo=a,grupo=g,real=r,prediccion=p,banco=banco,modelo=nombre)
                for a,g,r,p in zip(meta.archivo,grupos,y,pred))
            print(resumen[-1],flush=True)
    res=pd.DataFrame(resumen).sort_values(["porcentaje","porcentaje_total"],ascending=False,kind="stable")
    res.to_csv(salida/"resumen.csv",index=False)
    pd.DataFrame(registros).to_csv(salida/"predicciones_oof.csv",index=False)
    (salida/"auditoria.json").write_text(json.dumps(auditoria,indent=2),encoding="utf-8")
    meta.to_csv(salida/"muestra.csv",index=False)
    np.savez_compressed(salida/"bancos_desarrollo.npz",**bancos)
    print(res.head(10).to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos",type=Path,default=Path(F.RAIZ_PROYECTO)/"data2/CWRU")
    p.add_argument("--salida",type=Path,default=AQUI/"resultados_espectrales_desarrollo")
    a=p.parse_args(); a.salida.mkdir(parents=True,exist_ok=True)
    meta,bancos=extraer_desarrollo(a.datos,a.salida)
    comparar(meta,bancos,a.salida)
