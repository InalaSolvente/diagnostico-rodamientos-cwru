from pathlib import Path
import hashlib,json
import pandas as pd
from alinear_bandas import geometria
from predictor_espectral import MODELO


if __name__=="__main__":
    p=Path(__file__).parent/"resultados_L_alineado"
    d=pd.read_csv(p/"comparacion.csv")
    c=json.loads((p/"configuracion.json").read_text(encoding="utf-8"))
    assert len(d)==28 and d.archivo.is_unique
    assert d.L.eq(6).all() and d.ancho_final.eq(375).all()
    assert len(geometria(48000)[0])==31
    assert c['modelo_sha256']==hashlib.sha256(MODELO.read_bytes()).hexdigest()
    fallas=d[d.carpeta=='48k_DE']; sanos=d[d.carpeta=='Normal']
    assert len(fallas)==24 and fallas.correcto_ahora.sum()==12
    assert len(sanos)==4 and sanos.correcto_ahora.sum()==0
    assert d.pred_ahora.eq('IR').all()
    conclusion=("\n## Interpretación verificada\n\n"
        "El modelo asignó IR a las 28 señales: 12 aciertos en 24 fallas y ninguno en los cuatro normales. "
        "El 50% en fallas coincide con elegir siempre la clase IR; no demuestra discriminación entre tipos. "
        "Igualar anchos cambió las predicciones, pero no resolvió la transferencia a 48 kHz. "
        "Los pesos se entrenaron con la geometría anterior para los normales a 48k: "
        "una futura evaluación consistente requerirá preparar entrenamiento e inferencia con la misma representación.\n")
    ruta=p/'resumen.md'
    txt=ruta.read_text(encoding='utf-8').split('\n## Interpretación verificada')[0]
    ruta.write_text(txt+conclusion,encoding='utf-8')
    print('OK: L6, 375 Hz, 31 nodos fisicos, pesos intactos; 12/24 y 0/4; todas las predicciones IR')
