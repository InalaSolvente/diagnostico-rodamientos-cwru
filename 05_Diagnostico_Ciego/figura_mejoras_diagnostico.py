"""Transiciones del paper al grado conservador de la auditoría temporal."""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


if __name__ == '__main__':
    p = Path(__file__).parent
    d = pd.read_csv(p / 'auditoria_modelos_actuales/persistencia_temporal_sin_copias.csv')
    d = d.loc[d.fs.eq(12000)].copy()
    grados = ['ND', 'PD', 'D']
    columnas = grados + ['Tipo incorrecto']
    d['resultado'] = d.grado_conservador.where(d.tipo_correcto.eq(True), 'Tipo incorrecto')
    t = pd.crosstab(d.paper, d.resultado).reindex(
        index=grados, columns=columnas, fill_value=0)
    assert len(d) == 105 and d.archivo.is_unique
    assert d.paper.isin(grados).all() and d.grado_conservador.isin(grados).all()
    orden = {'ND': 0, 'PD': 1, 'D': 2}
    assert d.grado_conservador.map(orden).le(d.grado_original.map(orden)).all()
    assert t.to_numpy().sum() == 105
    mejoran = sum(int(t.iloc[i, j]) for i in range(3) for j in range(3) if j > i)
    iguales = sum(int(t.iloc[i, i]) for i in range(3))
    bajan = sum(int(t.iloc[i, j]) for i in range(3) for j in range(3) if j < i)
    errores = int(t['Tipo incorrecto'].sum())
    assert mejoran + iguales + bajan + errores == len(d)
    salida = p / 'resultados_modelos_todos'
    t.to_csv(salida / 'matriz_mejoras_auditada_105.csv')

    fig, ax = plt.subplots(figsize=(10, 6.7))
    colores = {'mejora': '#ccebd7', 'igual': '#dce8f4',
               'baja': '#f5d2cc', 'error': '#e1e1e1'}
    for i in range(3):
        for j in range(4):
            estado = 'error' if j == 3 else 'mejora' if j > i else 'baja' if j < i else 'igual'
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1,
                                  facecolor=colores[estado], edgecolor='white', linewidth=3))
            n = int(t.iloc[i, j])
            ax.text(j, i, f'{n}\n({100*n/t.iloc[i].sum():.1f} %)',
                    ha='center', va='center', fontsize=15, color='#172435')
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(2.5, -.5)
    ax.set_xticks(range(4), ['ND', 'PD', 'D', 'Tipo de falla\nincorrecto'])
    ax.set_yticks(range(3), grados)
    ax.set_xlabel('Grado conservador con comprobación temporal', fontsize=12)
    ax.set_ylabel('Clasificación del paper', fontsize=12)
    ax.tick_params(length=0, labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle('Transiciones de diagnóstico · comprobación temporal', fontsize=16, y=.96)
    ax.set_title('105 registros de 12 kHz con referencia · porcentaje por fila', fontsize=12, pad=18)
    etiquetas = [('mejora', f'Mejoran: {mejoran}'), ('igual', f'Iguales: {iguales}'),
                 ('baja', f'Empeoran: {bajan}'), ('error', f'Tipo incorrecto: {errores}')]
    ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=colores[k], label=v)
                       for k, v in etiquetas], loc='upper center', bbox_to_anchor=(.5, -.24),
              ncol=4, frameon=False, fontsize=10)
    fig.text(.5, .085, f'Sin retroceso y con tipo correcto: {mejoran+iguales}/105 ({100*(mejoran+iguales)/105:.2f} %)',
             ha='center', fontsize=12, weight='bold')
    fig.text(.5, .025, 'Tipo predicho con el registro completo: prueba condicional, no validación independiente.\n'
             'Grados propios, no homologados al paper. 48 kHz sin referencia transcrita.',
             ha='center', fontsize=9, linespacing=1.5)
    fig.subplots_adjust(left=.12, right=.97, top=.82, bottom=.31)
    fig.savefig(salida / 'matriz_mejoras_auditada_105.png', dpi=180)
    plt.close(fig)
    print(f'Mejoran={mejoran}; iguales={iguales}; empeoran={bajan}; tipo incorrecto={errores}')
