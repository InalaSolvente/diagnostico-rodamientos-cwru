# Comparación completa con/sin frecuencia esperada

133 registros cubiertos, sin reajustar modelos. Se conservan predicciones OOF de desarrollo.
Las matrices de cambios solo incluyen los 105 registros de 12 kHz con referencia del paper.
El total mezcla protocolos y condiciones compartidas; no es validación externa ni un ensayo de un único ajuste independiente.
Los normales no reciben un oráculo de salud: no tienen una frecuencia de falla esperada.
Las señales insuficientes se conservan como sin diagnóstico y no se eliminan del denominador.

48 kHz con falla: 21/24 tipos correctos sin frecuencia esperada; 1 sin duración suficiente.
Normales: 4/4 correctamente identificados sin frecuencia esperada.
48 kHz se evaluó con el mismo método jerárquico normalizado; no con el modelo histórico separado de 48 kHz.

[
  {
    "escenario": "Con frecuencia esperada · ORÁCULO",
    "total": 105,
    "mejoran": 21,
    "iguales": 74,
    "empeoran": 10,
    "tipo_incorrecto": 0,
    "sin_retroceso": 95,
    "porcentaje": 90.47619047619048
  },
  {
    "escenario": "Sin frecuencia esperada",
    "total": 105,
    "mejoran": 15,
    "iguales": 73,
    "empeoran": 7,
    "tipo_incorrecto": 10,
    "sin_retroceso": 88,
    "porcentaje": 83.80952380952381
  }
]
