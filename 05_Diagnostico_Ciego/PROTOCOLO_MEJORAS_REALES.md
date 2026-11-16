# Objetivo y protocolo de mejoras reales

Estado actual: se corrigió la selección de registros y RPM dentro de MAT con
múltiples adquisiciones. Los pilotos iniciales descritos más abajo son históricos;
no acreditan rendimiento de la base corregida. La entrada vigente está en
`resultados_desarrollo_verificado_v2/` y contiene 52 fallas y dos normales, con
canal, RPM, catálogo y hashes explícitos. Debe entrenarse y evaluarse de nuevo.
Ver `auditoria_selector_mat/INFORME.md`. La meta completa permanece pendiente.

Objetivo del usuario: muchas más mejoras que empeoramientos y al menos
90/105 sin retroceso, sin atajos. No se declara alcanzado por superar un
contador si falla la integridad o la verificación de evidencia.

## Etapa de desarrollo fijada antes de puntuar los candidatos

- Solo 52 fallas de desarrollo de 12 kHz y los dos normales de desarrollo.
  Las 53 fallas restantes no se abren para elegir cambios en esta etapa.
- Se conserva la regla de grado del usuario. No se eliminan picos de la izquierda,
  no se relajan sus exigencias ni se ajustan umbrales mirando el paper.
- No se introducen el tipo real ni la frecuencia real esperada en inferencia.
  Solo señal, muestreo, sensor configurado y catálogo completo con RPM disponibles.
- Tipos y grados incorrectos siguen separados; una equivocación no se convierte en ND.
- La corrección nominal entre pruebas no se presenta como tasa calibrada de falsas alarmas.

## Dos mejoras concretas

1. `bandas_estables.py`: compara todos los candidatos de selección, con Hann,
   extensión reflejada y envolvente de amplitud/cuadrada. Prioriza márgenes
   sostenidos en subsegmentos. La banda, familia y transformación se fijan antes
   de ver la otra mitad. Esta última no permite buscar una alternativa favorable.
2. `ensayar_tipo_seleccion.py`: una receta fija de clasificación, con prefijo
   de 2 s para todas las señales. Los normales de 48 kHz se remuestrean con
   antialias a 12 kHz. Igualar fs, duración y bandas elimina esos indicadores
   triviales de adquisición, aunque no demuestra eliminar todos los factores
   de confusión. Escalado y ajuste se realizan dentro de cada pliegue.

Los JSON de cada experimento fijan receta y hash de código antes de producir
predicciones. Un cambio requiere otro experimento, no reescribir el anterior.

## Interpretación y pendientes obligatorios

El primer piloto de bandas mantiene tipos históricos que utilizaron el registro
completo: es condicional y no puede acreditar el objetivo final. El segundo
candidato intenta resolver ese acceso al segmento de verificación.

Antes de una evaluación total se debe congelar una receta elegida con desarrollo,
medir también errores en controles normales y sintéticos, resolver la incertidumbre
de RPM de la copia normal y asegurar grupos sin señales repetidas. Dos normales
de desarrollo no permiten certificar generalización a rodamientos sanos nuevos.

Si se comparan variantes por rendimiento, su elección debe realizarse dentro de
los pliegues de entrenamiento o en desarrollo, nunca mirando la reserva para
retocar la receta. La reserva actual fue observada históricamente: una nueva
ejecución congelada será una comprobación interna, no una validación externa.

Se informarán mejoras, iguales, empeoramientos y errores de tipo por separado.
Los grados propios no se presentarán como una homologación de la rúbrica del paper.
El mínimo 90/105 debe comprobarse en el total de referencia y con las salvaguardas,
no extrapolarse a partir de una buena puntuación en desarrollo.
