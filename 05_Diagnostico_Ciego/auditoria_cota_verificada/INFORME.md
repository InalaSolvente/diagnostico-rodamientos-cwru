# Cota con verificación y tipo conocido

Esto es una auditoría con etiquetas, no el rendimiento de un modelo ciego.
Se concede al método un clasificador y un detector perfectos, pero se conservan el catálogo
completo del fabricante, el selector de dos bandas, las transformaciones y el juez actuales.
La primera mitad selecciona las bandas; la segunda verifica esas mismas bandas. No se
escoge retrospectivamente otra banda ni se combina evidencia de ambas para fabricar armónicos.

 banco  total  cota_sin_retroceso  mejoras_con_tipo_conocido
    31    105                  97                         23
    57    105                  99                         24
    63    105                  97                         23

La cota es específica de cada selector fijo: un método distinto de selección, otro banco
o un preprocesado legítimo podrían cambiarla. No demuestra una imposibilidad universal.
Si una cota queda por debajo de 100, mejorar solo el clasificador no alcanza la meta con
ese selector. Los resultados no se incorporan a las predicciones ni a sus matrices.

Casos que retroceden aun concediendo el tipo verdadero:

              archivo  banco paper grado_con_tipo_conocido
    12k_DE/B007_3.mat     31     D                      PD
    12k_DE/B007_3.mat     63     D                      PD
    12k_DE/B007_3.mat     57     D                      PD
    12k_DE/B021_0.mat     31     D                      PD
    12k_DE/B021_0.mat     63     D                      PD
    12k_DE/B021_0.mat     57     D                      PD
    12k_DE/B021_1.mat     31     D                      PD
    12k_DE/B021_1.mat     63     D                      PD
    12k_DE/B021_1.mat     57     D                      PD
    12k_DE/B028_0.mat     31     D                      PD
    12k_DE/B028_0.mat     63     D                      PD
 12k_DE/OR014@6_0.mat     31     D                      PD
 12k_DE/OR014@6_0.mat     63     D                      PD
 12k_DE/OR014@6_0.mat     57     D                      PD
12k_DE/OR021@12_0.mat     31     D                      PD
12k_DE/OR021@12_0.mat     63     D                      PD
12k_DE/OR021@12_2.mat     31     D                      PD
12k_DE/OR021@12_2.mat     63     D                      PD
12k_DE/OR021@12_2.mat     57     D                      PD
   12k_FE/IR007_1.mat     31     D                      PD
   12k_FE/IR007_1.mat     63     D                      PD
   12k_FE/IR007_1.mat     57     D                      PD
