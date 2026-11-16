# Criterio del usuario: todos los registros

D: primera identificada, al menos dos armónicas, ningún pico previo mayor; los picos a la derecha no impiden D.
PD: una o más armónicas identificadas y un pico previo mayor; la primera puede faltar.
Si falta h=1, la comparación PD se hace contra la primera armónica realmente identificada, nunca contra ruido en la fundamental.
ND: todo lo demás. Se excluye DC. Armónicas: picos locales dentro de ±2%, significativos contra el fondo local.
Se buscan hasta seis armónicas por hipótesis, con la corrección nominal documentada en configuracion.json.
La primera es respecto a la frecuencia base de la familia declarada (BSF y 2BSF son hipótesis explícitas diferentes).
El tipo de falla permanece congelado: OOF en desarrollo; modelo ajustado en desarrollo para reserva y adicionales.
Un tipo incorrecto con PD/D se informa como tipo_incorrecto; nunca se transforma a ND para sumar un acierto.
Estos grados no se presentan como homologación de la rúbrica del paper.

Señales ejecutadas: 133. Con referencia del paper: 105. Sin retroceso estricto: 90/105.
Grados emitidos en fallas: {'D': 75, 'PD': 30}
Grados emitidos en normales: {'PD': 2, 'ND': 2}
Grados emitidos en adicionales 48k sin referencia transcrita: {'ND': 23, 'PD': 1}
Los normales y adicionales 48k no se confunden con ND del paper ni se suman al porcentaje de no retroceso.
La comparación del paper mezcla protocolos de desarrollo/reserva y no es validación externa independiente.
No hubo entrenamiento ni ajuste de parámetros con estos resultados.
