"""Calidad ordinal frente a S&R; no equivale a exactitud de tipo de falla."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F, validar_catalogo
from criterio_usuario import comparar as comparar_grado

RAIZ=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(RAIZ/"02_Sinergia_Fraterna/codigo"))
from experimento_cwru import BENCH

ORDEN={"ND":0,"PD":1,"D":2}
PAPER={"N":"ND","P":"PD","Y":"D"}


def calificar(x,fs,catalogo,tipo_predicho,L=4,p=.01,*,criterio="historico",nodos=None):
    """Calidad operativa del tipo predicho, sin consultar defecto real ni paper."""
    validar_catalogo(catalogo,fs)
    if criterio not in ("historico","usuario"):
        raise ValueError("Criterio inválido")
    if nodos is not None:
        nodos=set(tuple(n) for n in nodos)
        if not nodos or any(not 0<=l<=L or not 0<=m<2**l for l,m in nodos):
            raise ValueError("Nodos inválidos")
    if tipo_predicho not in ("IR","OR","B","FTF"):
        return dict(grado="ND",nodo=None,familia=None)
    familias={"IR":[("BPFI",catalogo["BPFI"])],"OR":[("BPFO",catalogo["BPFO"])],
              "B":[("BSF",catalogo["BSF"]),("2BSF",2*catalogo["BSF"])],
              "FTF":[("FTF",catalogo["FTF"])]}
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all() or len(x)<1024 or not np.isfinite(fs) or fs<=0:
        raise ValueError("Señal o muestreo inválidos")
    n=len(x); X=np.fft.fft(x-x.mean())
    f_abs=abs(np.fft.fftfreq(n,1/fs))
    hilbert=np.zeros(n); hilbert[0]=1; hilbert[1:(n+1)//2]=2
    if n%2==0: hilbert[n//2]=1
    alpha=np.fft.rfftfreq(n,1/fs)
    # Se corrigen 31 bandas y todas las familias, no solo la elegida por el modelo.
    H=6 if criterio=="usuario" else 3
    cantidad=len(nodos) if nodos is not None else 2**(L+1)-1
    p_prueba=p/(cantidad*5*H)
    mejor=dict(grado="ND",nodo=None,familia=None,p_prueba=p_prueba)
    for l in range(L+1):
        ancho=fs/2**(l+1)
        for m in range(2**l):
            if nodos is not None and (l,m) not in nodos: continue
            mascara=(f_abs>=m*ancho)&((f_abs<=(m+1)*ancho) if m==2**l-1 else (f_abs<(m+1)*ancho))
            a=abs(np.fft.ifft(X*mascara*hilbert)); a-=a.mean()
            s=abs(np.fft.rfft(a))**2/n**2
            if not s.any(): continue
            for familia,f in familias[tipo_predicho]:
                if criterio=="usuario":
                    from criterio_usuario import evaluar
                    diag=evaluar(alpha,s,f,H=H,p=p_prueba)
                    grado=diag["grado"]
                else:
                    diag=F.evaluar_diagnostico(alpha,s,f,H=3,p=p_prueba)
                    grado={"N":"ND","P":"PD","D":"D"}[diag["categoria"]]
                if (ORDEN[grado]>ORDEN[mejor["grado"]] or
                    (criterio=="usuario" and grado==mejor["grado"] and
                     len(diag["identificadas"])>len(mejor.get("evidencia",{}).get("identificadas",[]))) or
                    (criterio=="usuario" and "evidencia" not in mejor)):
                    mejor=dict(grado=grado,nodo=[l,m],familia=familia,p_prueba=p_prueba)
                    if criterio=="usuario": mejor["evidencia"]=diag
            if mejor["grado"]=="D": return mejor
    return mejor


def ejecutar():
    aqui=Path(__file__).parent
    salida=aqui/"resultados_no_retroceso_desarrollo"
    salida.mkdir(exist_ok=True)
    pred=pd.read_csv(aqui/"resultados_espectrales_desarrollo/verificacion_oof.csv")
    particion=pd.read_csv(aqui/"particion_desarrollo.csv")
    dev=set(particion.loc[particion.particion=="desarrollo","archivo"])
    if set(pred.archivo)!=dev:
        raise ValueError("Las predicciones deben ser las de desarrollo fuera de grupo")
    datos=RAIZ/"data2/CWRU"
    idx=F.leer_indice(str(datos)).set_index("archivo")
    filas=[]
    for _,r in pred.iterrows():
        meta=idx.loc[r.archivo]
        if meta.carpeta=="Normal": continue
        if int(meta.registro) not in BENCH: raise ValueError("Registro sin referencia S&R")
        mat=loadmat(datos/r.archivo)
        clave=F.clave_canal(mat, meta.sensor_falla, int(meta.registro))
        claves_rpm=[k for k in mat if k.endswith("RPM")]
        rpm=F.rpm_registro(mat, int(meta.registro), nominal=float(meta.rpm_nominal))
        catalogo={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[meta.sensor_falla].items()}
        calidad=calificar(np.ravel(mat[clave]),float(meta.fs),catalogo,r.prediccion)
        # Únicamente después de inferir/calificar se consulta la referencia real.
        referencia=PAPER[BENCH[int(meta.registro)]]
        tipo_correcto=r.prediccion==r.real
        grado,exito=comparar_grado(referencia,calidad["grado"],tipo_correcto)
        filas.append(dict(archivo=r.archivo,registro=int(meta.registro),paper=referencia,
                          prediccion=r.prediccion,tipo_real=r.real,tipo_correcto=tipo_correcto,
                          grado_emitido=calidad["grado"],grado_evaluable=grado,sin_retroceso=exito))
        print(f"{len(filas)}/52 {meta.registro}: {referencia} -> {grado}",flush=True)
    df=pd.DataFrame(filas)
    df.to_csv(salida/"comparacion.csv",index=False)
    tabla=pd.crosstab(df.paper,df.grado_evaluable).reindex(index=ORDEN,columns=[*ORDEN,'tipo_incorrecto'],fill_value=0)
    tabla.to_csv(salida/"transiciones.csv")
    lineas=["# No retroceso frente al paper: desarrollo", "",
        f"Sin retroceso: {int(df.sin_retroceso.sum())}/{len(df)} ({100*df.sin_retroceso.mean():.2f}%).",
        f"Tipo correcto: {int(df.tipo_correcto.sum())}/{len(df)}.",
        "Referencia: categorías S&R transcritas en el proyecto (Y=D, P=PD, N=ND).",
        "Grado operativo: criterio espectral existente, explorando los 31 nodos para el tipo predicho y corrigiendo por bandas, familias y armónicas.",
        "La clasificación de tipo se fijó fuera de grupo. La calidad no recibe fd real ni clasificación del paper.",
        "Si el tipo es incorrecto se evalúa como ND; las filas ND del paper no pueden empeorar y cuentan automáticamente.",
        "No retroceso es una métrica ordinal, no exactitud diagnóstica. No es una reproducción del algoritmo del paper.",
        "Los 55 archivos reservados permanecen excluidos."]
    (salida/"resumen.md").write_text("\n".join(lineas)+"\n",encoding="utf-8")
    print("\n".join(lineas))


if __name__=="__main__":
    ejecutar()
