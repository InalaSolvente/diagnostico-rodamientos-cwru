# Preblanqueo AR(32): ensayo descartado

Resultado auditado: {'total': 105, 'mejoran': 9, 'iguales': 64, 'empeoran': 2, 'tipo_incorrecto': 30, 'sin_retroceso': 73, 'normales_correctos': 3}. La referencia estricta sin AR conserva 79/105 y cuatro normales correctos.
Se mantienen los 133 registros y las 105 referencias. Dos vistas FE adicionales se usan solo para entrenar el detector: no aumentan el número de casos.
Se comprobó la exclusión conjunta de posiciones OR y de las vistas de cada normal. El filtro no recibe catálogo ni etiquetas; usa los primeros 2,5 segundos y continúa causalmente.
Las pruebas sintéticas y dos registros reales confirmaron que reemplazar la segunda mitad no altera filtro ni selección. Se verificaron las reglas D por rama, sin fusionar armónicas.
El filtro reduce empeoramientos de grado, pero incrementa errores de tipo y falla en un normal. No se adopta ni se reemplazan modelos anteriores.
El objetivo activo es 100/105 sin retroceso y muchas más mejoras que empeoramientos. No se alcanzó. El ensayo sigue siendo interno sobre datos ya inspeccionados.
Verificar: `python 05_Diagnostico_Ciego/verificar_preblanqueo_ar.py`.
