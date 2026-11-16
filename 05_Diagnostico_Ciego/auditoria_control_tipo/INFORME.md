# Límite estructural del control de tipo

Se enumeran los tres posibles resultados del clasificador, sin utilizar la etiqueta para elegir ninguno. Después se comprueba qué tipos permite emitir cada controlador.
Esto es una auditoría de alcanzabilidad, no una predicción ni un resultado oracular presentado como modelo.

        control  techo_estricto  techo_ND_emitido_optimista  total                                      tipo
sin_forzar_tipo             100                         100    105 cota optimista, no desempeño de un modelo
      control_D             100                         100    105 cota optimista, no desempeño de un modelo
 control_D_y_PD              98                          98    105 cota optimista, no desempeño de un modelo

Casos adicionales que el controlador completo impide conservar, aunque el clasificador inicialmente identificara bien el elemento:

          archivo paper tipo_real_solo_auditoria max_grado_seleccion alcanzables_D alcanzables_D_PD  sin_control_posible  con_D_posible  con_D_PD_posible  control_impide_tipo_real
12k_DE/B014_0.mat    PD                        B                  PD       B;IR;OR               IR                 True           True             False                      True
12k_DE/B014_2.mat    PD                        B                  PD       B;IR;OR               OR                 True           True             False                      True

La cota combina el tipo alcanzable y el mejor grado de selección del banco31. La verificación solo puede mantener o reducir ese grado, por lo que la cifra es optimista.
Para ND se muestra también una cota que permite cualquier ND efectivamente emitido: no se modifica el conteo de las matrices ni se inventan diagnósticos.
La conclusión corresponde a estas reglas y este banco sobre señal cruda; no es un límite universal del dataset ni de otros filtros.
No se excluye ningún registro y estas listas no son entradas de ningún predictor.

Reproducir: `python 05_Diagnostico_Ciego/auditar_control_tipo.py`.
