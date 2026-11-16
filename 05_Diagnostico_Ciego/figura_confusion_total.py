"""Matriz por tipo de falla; excluye registros sin predicción evaluable."""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    p = Path(__file__).parent / 'resultados_modelos_todos'
    datos = pd.read_csv(p / 'consolidado_133.csv')
    d = datos.loc[datos.evaluable.eq(True)].copy()
    clases = ['B', 'IR', 'OR', 'N']
    nombres = ['Bola', 'Pista interna', 'Pista externa', 'Normal']
    assert len(datos) == 133 and len(d) == 129 and datos.archivo.is_unique
    assert d.real.isin(clases).all() and d.prediccion.isin(clases).all()
    matriz = pd.crosstab(d.real, d.prediccion).reindex(
        index=clases, columns=clases, fill_value=0)
    correctos = int(matriz.to_numpy().trace())
    assert matriz.to_numpy().sum() == len(d)
    assert correctos == int(d.tipo_correcto.sum())
    matriz.to_csv(p / 'matriz_confusion_auditada.csv')

    fig, ax = plt.subplots(figsize=(8.5, 7.4))
    t = matriz.to_numpy()
    ax.imshow(t, cmap='Blues', vmin=0, vmax=t.max())
    for i in range(len(clases)):
        for j in range(len(clases)):
            porcentaje = 100 * t[i, j] / t[i].sum()
            ax.text(j, i, f'{t[i, j]}\n({porcentaje:.1f} %)', ha='center',
                    va='center', fontsize=12,
                    color='white' if t[i, j] > t.max() / 2 else 'black')
    ax.set_xticks(range(4), nombres)
    ax.set_yticks(range(4), nombres)
    ax.set_xlabel('Tipo predicho', fontsize=12)
    ax.set_ylabel('Tipo real', fontsize=12)
    ax.set_title(f'{correctos}/{len(d)} correctos ({100 * correctos / len(d):.2f} %)\n'
                 'Celdas: cantidad y porcentaje del tipo real', fontsize=12, pad=15)
    fig.suptitle('Matriz de confusión · modelos de 12 y 48 kHz', fontsize=15, y=.96)
    fig.text(.5, .055,
             '133 archivos: 129 evaluables; excluidos 2 normales de entrenamiento y 2 copias.\n'
             'Combina validación cruzada y reserva con condiciones compartidas.\n'
             'No es validación externa independiente; evalúa tipo, no grado ND/PD/D.',
             ha='center', fontsize=10, linespacing=1.5)
    fig.subplots_adjust(left=.18, right=.96, bottom=.21, top=.82)
    fig.savefig(p / 'matriz_confusion_auditada.png', dpi=180)
    plt.close(fig)
    print(matriz.to_string())
    print(f'Correctos: {correctos}/{len(d)}; excluidos: {len(datos)-len(d)}')
