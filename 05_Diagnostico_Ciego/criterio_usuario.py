"""Regla ND/PD/D basada en armónicas identificadas y dominancia de la primera."""
import numpy as np

ORDEN={"ND":0,"PD":1,"D":2}


def evaluar(alpha,s,frecuencia,*,H=6,tol=.02,p=.01):
    alpha=np.asarray(alpha,dtype=float); s=np.asarray(s,dtype=float)
    if alpha.ndim!=1 or s.shape!=alpha.shape or len(alpha)<3:
        raise ValueError("Se requiere un espectro unidimensional con al menos tres bins")
    if not np.isfinite(alpha).all() or not np.isfinite(s).all() or np.any(s<0) or np.any(alpha<0) or np.any(np.diff(alpha)<=0):
        raise ValueError("Frecuencias ordenadas y potencia finita no negativa requeridas")
    if not isinstance(H,int) or H<2 or not np.isfinite([frecuencia,tol,p]).all() or frecuencia<=0 or not 0<tol<1/(2*H) or not 0<p<1:
        raise ValueError("Parámetros de armónicas inválidos")
    armonicas=[]
    for h in range(1,H+1):
        distancia=abs(alpha-h*frecuencia)
        ventana=(distancia<=tol*h*frecuencia)&(alpha>0)
        fondo=(distancia<=.3*frecuencia)&~ventana&(alpha>0)
        if not ventana.any() or not fondo.any(): continue
        i=int(np.flatnonzero(ventana)[np.argmax(s[ventana])])
        M=int(ventana.sum())
        umbral=float(-np.log(-np.expm1(np.log1p(-p)/M)))
        media=float(s[fondo].mean()); potencia=float(s[i])
        pico=0<i<len(s)-1 and s[i]>s[i-1] and s[i]>=s[i+1]
        significativa=bool(pico and potencia>0 and potencia>umbral*media)
        armonicas.append(dict(h=h,frecuencia_hz=float(alpha[i]),potencia=potencia,
                              fondo=media,umbral=umbral,significativa=significativa))
    identificadas=[a for a in armonicas if a["significativa"]]
    primera=next((a for a in armonicas if a["h"]==1),None)
    primera_valida=bool(primera and primera["significativa"])
    potencia_primera=primera["potencia"] if primera else 0.
    # Si falta h=1, PD se compara con la primera armónica realmente identificada,
    # no con ruido situado en la ventana vacía de la fundamental.
    referencia=primera if primera_valida else (identificadas[0] if identificadas else primera)
    potencia_referencia=referencia["potencia"] if referencia else 0.
    h_referencia=referencia["h"] if referencia else 1
    izquierda=(alpha>0)&(alpha<h_referencia*frecuencia*(1-tol))
    max_izquierda=float(s[izquierda].max()) if izquierda.any() else 0.
    i_izquierda=int(np.flatnonzero(izquierda)[np.argmax(s[izquierda])]) if izquierda.any() else None
    mayor_izquierda=max_izquierda>potencia_referencia and not np.isclose(max_izquierda,potencia_referencia,rtol=1e-12,atol=0)
    max_total=float(s[alpha>0].max())
    primera_maxima=max_total<=potencia_primera or np.isclose(max_total,potencia_primera,rtol=1e-12,atol=0)
    grado="ND"; motivo="menos_de_dos_armonicas"
    if len(identificadas)>=2 and primera_valida and not mayor_izquierda:
        grado="D"; motivo="primera_dominante_a_izquierda_y_dos_o_mas_armonicas"
    elif len(identificadas)>=1 and mayor_izquierda:
        grado="PD"; motivo="una_o_mas_armonicas_con_pico_previo_mayor"
    elif len(identificadas)>=2 and not primera_valida:
        motivo="primera_no_identificada_sin_pico_previo_mayor"
    return dict(grado=grado,motivo=motivo,frecuencia_base=float(frecuencia),armonicas=armonicas,
        identificadas=[a["h"] for a in identificadas],primera_identificada=primera_valida,
        potencia_primera=potencia_primera,max_izquierda=max_izquierda,
        h_referencia_izquierda=h_referencia,potencia_referencia_izquierda=potencia_referencia,
        frecuencia_max_izquierda=float(alpha[i_izquierda]) if i_izquierda is not None else None,
        hay_pico_previo_mayor=bool(mayor_izquierda),primera_maxima_total=bool(primera_maxima),
        dominancia="solo_izquierda",min_armonicas_d=2,min_armonicas_pd=1)


def comparar(referencia,grado,tipo_correcto):
    if referencia not in ORDEN or grado not in ORDEN:
        raise ValueError("Grado inválido")
    # ND cuenta solo si fue emitido de verdad, nunca fabricado por la etiqueta real.
    if grado=="ND": return "ND",referencia=="ND"
    if not tipo_correcto: return "tipo_incorrecto",False
    return grado,ORDEN[grado]>=ORDEN[referencia]
