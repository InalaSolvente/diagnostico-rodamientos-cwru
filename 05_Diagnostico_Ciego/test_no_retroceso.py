from evaluar_no_retroceso import comparar_grado,ORDEN
from pathlib import Path
import hashlib
import json
import pandas as pd


if __name__=="__main__":
    for anterior in ORDEN:
        for ahora in ORDEN:
            grado,exito=comparar_grado(anterior,ahora,True)
            assert exito==(ORDEN[ahora]>=ORDEN[anterior])
            assert grado==ahora
            grado,exito=comparar_grado(anterior,ahora,False)
            assert grado==('ND' if ahora=='ND' else 'tipo_incorrecto')
            assert exito==(anterior=='ND' and ahora=='ND')
    assert comparar_grado("D","PD",True)==("PD",False)
    assert comparar_grado("PD","ND",True)==("ND",False)
    assert comparar_grado("ND","D",True)==("D",True)
    aqui=Path(__file__).parent
    ruta=aqui/"resultados_paper_total/comparacion.csv"
    if ruta.exists():
        df=pd.read_csv(ruta)
        assert len(df)==105 and df.archivo.is_unique and df.registro.is_unique
        resultados=[]
        for r in df.itertuples():
            grado,exito=comparar_grado(r.paper,r.grado_emitido,r.tipo_correcto)
            resultados.append(exito)
            if not r.tipo_correcto and r.grado_emitido!='ND':
                assert grado=='tipo_incorrecto' and not exito
        # La evaluación se deriva de salidas emitidas, no del contador histórico.
        assert sum(resultados)==sum(
            (r.paper=='ND' and r.grado_emitido=='ND') or
            (r.tipo_correcto and ORDEN[r.grado_emitido]>=ORDEN[r.paper])
            for r in df.itertuples())
        assert df.tipo_correcto.sum()==98
        assert (df.origen=="desarrollo_oof").sum()==52 and (df.origen=="reserva_congelado").sum()==53
        config=json.loads((ruta.parent/"configuracion.json").read_text())
        assert config['modelo_sha256']==hashlib.sha256((aqui/'resultados_espectrales_desarrollo/modelo.npz').read_bytes()).hexdigest()
    print("OK: nueve transiciones; tipo incorrecto nunca convertido en ND para sumar aciertos")
