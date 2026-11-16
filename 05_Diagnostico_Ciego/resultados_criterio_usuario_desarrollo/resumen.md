# Criterio del usuario: desarrollo

D: primera identificada, al menos dos armónicas, ningún pico previo mayor; los picos a la derecha no impiden D.
PD: una o más armónicas identificadas y un pico previo mayor; la primera puede faltar.
Si falta h=1, la comparación PD se hace contra la primera armónica realmente identificada, nunca contra ruido en la fundamental.
ND: todo lo demás. Se excluye DC. Armónicas: picos locales dentro de ±2%, significativos contra el fondo local.
Se buscan hasta seis armónicas por hipótesis, con la corrección nominal documentada en configuracion.json.
La primera es respecto a la frecuencia base de la familia declarada (BSF y 2BSF son hipótesis explícitas diferentes).
El tipo de falla permanece congelado: procede de las predicciones fuera de grupo ya guardadas.
Un tipo incorrecto con PD/D se informa como tipo_incorrecto; nunca se transforma a ND para sumar un acierto.
Estos grados no se presentan como homologación de la rúbrica del paper.

Señales de desarrollo: 54. Con falla: 52. Sin retroceso estricto: 40/52.
Grados emitidos en fallas: {'D': 35, 'PD': 17}
Grados emitidos en normales: {'PD': 2}
Los demás archivos no se ejecutan. No hubo entrenamiento ni ajuste con estos resultados.
