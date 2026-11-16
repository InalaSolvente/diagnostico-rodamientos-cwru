# Dos formas de contar un ND ya emitido

La función original `criterio_usuario.comparar` cuenta ND→ND como acierto sin exigir el tipo interno. Varias matrices posteriores usaron tipo correcto como condición para todo acierto, incluido ND.
Este informe calcula ambos criterios sobre los mismos diagnósticos. No transforma un PD/D equivocado en ND, no modifica etiquetas y no vuelve a entrenar modelos.
La elección del criterio para la meta de 100/105 se consultó al usuario y permanece pendiente. Se conserva el conteo estricto como referencia provisional.
Los resultados desfavorables incluyen tanto disminuciones de grado como tipos incorrectos. Comparar solo mejoras contra disminuciones de grado ocultaría diagnósticos erróneos.
Los protocolos con posiciones OR separadas se etiquetan aparte y no deben presentarse como equivalentes al control que excluye todas las posiciones juntas.

                     modelo   agrupacion                   criterio  total  mejoran  iguales  empeoran  tipo_incorrecto  sin_retroceso  desfavorables_total  razon_mejoras_desfavorables
referencia_grupos_completos    OR_juntas         tipo_exigido_en_ND    105       13       66         4               22             79                   26                     0.500000
referencia_grupos_completos    OR_juntas ND_emitido_sin_diagnostico    105       13       66         4               22             79                   26                     0.500000
           preblanqueo_AR32    OR_juntas         tipo_exigido_en_ND    105        9       64         2               30             73                   32                     0.281250
           preblanqueo_AR32    OR_juntas ND_emitido_sin_diagnostico    105        9       65         3               28             74                   31                     0.290323
  curvas_coherentes_LDA_SVM    OR_juntas         tipo_exigido_en_ND    105        8       65         5               27             73                   32                     0.250000
  curvas_coherentes_LDA_SVM    OR_juntas ND_emitido_sin_diagnostico    105        8       65         5               27             73                   32                     0.250000
     curvas_menu_con_ranker    OR_juntas         tipo_exigido_en_ND    105        4       63         4               34             67                   38                     0.105263
     curvas_menu_con_ranker    OR_juntas ND_emitido_sin_diagnostico    105        4       63         4               34             67                   38                     0.105263
        curvas_y_modulacion    OR_juntas         tipo_exigido_en_ND    105        9       61         5               30             70                   35                     0.257143
        curvas_y_modulacion    OR_juntas ND_emitido_sin_diagnostico    105        9       61         5               30             70                   35                     0.257143
      protocolo_anterior_88 OR_separadas         tipo_exigido_en_ND    105       16       72         5               12             88                   17                     0.941176
      protocolo_anterior_88 OR_separadas ND_emitido_sin_diagnostico    105       16       72         5               12             88                   17                     0.941176

Fuente del criterio: [criterio_usuario.py](../criterio_usuario.py).
Reproducir: `python 05_Diagnostico_Ciego/auditar_conteo_nd.py`.
