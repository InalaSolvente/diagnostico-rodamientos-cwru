# Auditoría agrupando las posiciones OR

Resultado con todas las posiciones del mismo tipo/tamaño excluidas juntas: {'total': 105, 'mejoran': 13, 'iguales': 66, 'empeoran': 4, 'tipo_incorrecto': 22, 'sin_retroceso': 79}.
Resultado previo, distinguiendo posición: {'total': 105, 'mejoran': 16, 'iguales': 72, 'empeoran': 5, 'tipo_incorrecto': 12, 'sin_retroceso': 88}.
Se reentrenaron tanto el detector como el tipo sin el grupo completo; únicamente se usaron los 52 casos de desarrollo y las dos señales normales de entrenamiento. Las reglas y el banco31 se fijaron antes de esta auditoría.
Se verificó que los casos ajenos a OR conservan exactamente su tipo y grado. La comparación mantiene los mismos 133 archivos, los 105 con referencia y todas las señales insuficientes.

## Interpretación metodológica

Agrupar por tipo y tamaño es una precaución conservadora: no demuestra que cada grupo equivalga a un rodamiento físico distinto.
CWRU documenta experimentos con distintas posiciones OR y explica que la posición afecta la vibración. No publica aquí identificadores suficientes para demostrar independencia entre todos los rodamientos.
Fuente: [CWRU, aparato y procedimientos](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures) y [tabla de datos DE](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data).
La misma página indica SKF para defectos 7/14/21 mil y NTN equivalentes para 28/40 mil. El catálogo configurado usa las razones DE/FE publicadas para SKF; faltan coeficientes NTN específicos para confirmar esa equivalencia, y no se deben estimar conociendo el defecto verdadero.
Las reglas también se desarrollaron tras inspeccionar el dataset. Este control no elimina el sesgo por selección reiterada ni sustituye validación externa.
Los p siguen siendo nominales; el desempeño sobre cuatro normales no acredita una tasa poblacional de falsas alarmas.

La meta sigue siendo al menos 90/105 sin retroceso, con muchas más mejoras que empeoramientos. No se cambia el denominador ni se convierten errores de tipo en ND usando la etiqueta real.
Verificar: `python 05_Diagnostico_Ciego/verificar_grupos_por_defecto.py`.
