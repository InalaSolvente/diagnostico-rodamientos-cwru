# Modelo propio de 48 kHz

Mismo clasificador LDA regularizado y representación alineada: L=6, 31 nodos 0–6 kHz y bandas finales de 375 Hz.
Entrenamiento: 14 señales (6 bola, 6 interna, 2 normales); las otras 14 no se cargaron para este entrenamiento/ensayo.
Primera opción correcta fuera de condición: 10/12 (83.33%).
Seis grupos de falla excluidos por turno. Escalado y ajuste solo en entrenamiento; sin selección de variantes contra los resultados.
Los dos normales permanecen en entrenamiento: no existe otro grupo normal para validarlos fuera de grupo. No se informa exactitud en normales.
No hay ejemplos de pista externa o jaula en este subconjunto: esas clases no se entrenan ni se declaran cubiertas.
La reserva comparte condiciones con desarrollo y fue examinada en experimentos anteriores; no es un conjunto externo intacto.
El modelo final se ajusta con las 14 señales. Los porcentajes anteriores no son aciertos de entrenamiento.
Inferencia: señal de 48 kHz y catálogo completo; sin frecuencia ni defecto real esperado.
El árbol conserva solo 0–6 kHz para mantener la representación alineada; no utiliza 6–24 kHz.
