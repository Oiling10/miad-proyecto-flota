"""
Figuras del informe de la Guía 4 (se guardan en docs/figuras/).

Requiere haber ejecutado antes:
    python src/anonimizar.py        (solo quien tenga data/raw/)
    python src/preprocesamiento.py

Uso:
    python src/figuras.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from src import eda  # noqa: E402

PROC = BASE / "data" / "procesados"
ANON = BASE / "data" / "anonimizados"
FIG = BASE / "docs" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)


def guardar(fig, nombre):
    fig.savefig(FIG / nombre, dpi=160, bbox_inches="tight")
    print(f"  docs/figuras/{nombre}")


def main():
    eda.configurar_estilo()
    c = eda.colores()
    m = pd.read_csv(PROC / "mantenimientos_limpios.csv", parse_dates=["fecha"])
    diario = pd.read_csv(PROC / "tabla_diaria.csv", parse_dates=["fecha"])
    ev_raw = eda.leer_eventos(ANON / "Detailed Event Report_anonimizado.csv")
    ev, _ = eda.convertir_tipos_eventos(ev_raw)

    # 1. Voltaje diario por camioneta
    guardar(eda.grafico_voltaje_diario(ev), "01_voltaje_diario.png")

    # 2. Caso de batería: camionetas con cambio de batería dentro del periodo de telemetría
    bat = m[m["es_correctivo"] & m["descripcion"].str.contains("BATER", case=False)]
    for placa, g in bat.groupby("placa"):
        fechas = [f for f in g["fecha"] if f >= ev["inicio"].min()]
        if fechas:
            guardar(eda.grafico_bateria_camioneta(ev, placa, fechas_cambio=fechas), f"02_bateria_{placa}.png")

    # 3. km por día según los viajes (sin uso vs. sin dato)
    vd = diario.dropna(subset=["estado_dia"]) if "estado_dia" in diario else None
    if vd is not None and len(vd):
        guardar(eda.grafico_km_dia(vd), "03_km_dia_viajes.png")

    # 4. Uso frente a alertas de batería
    perfil = eda.uso_vs_bateria(diario)
    perfil.to_csv(BASE / "docs" / "uso_vs_bateria.csv")
    guardar(eda.grafico_uso_vs_bateria(perfil), "04_uso_vs_bateria.png")

    # 5. Sensibilidad de la etiqueta "no programado" al umbral de días
    corr = m[m["es_correctivo"]]
    umbrales = [0, 1, 3, 5, 7, 15]
    pct = [(corr["dias_a_preventivo"] > u).mean() * 100 for u in umbrales]
    fig, a = plt.subplots(figsize=(6, 3.2))
    barras = a.bar([str(u) for u in umbrales], pct, color=[c["acento"] if u == 3 else c["gris"] for u in umbrales])
    a.bar_label(barras, labels=[f"{p:.0f} %" for p in pct], fontsize=9)
    a.set_xlabel("Umbral: días entre el correctivo y el preventivo más cercano")
    a.set_ylabel("% de correctivos 'no programados'")
    a.set_title(f"Sensibilidad de la etiqueta al umbral · {len(corr)} correctivos · rojo: criterio de la Guía 2")
    fig.tight_layout()
    guardar(fig, "05_sensibilidad_etiqueta.png")
    plt.close(fig)

    # 6. Correctivos por sistema (etiquetas múltiples)
    cols = [col for col in m.columns if col.startswith("corr_")]
    sis = corr[cols].sum().sort_values()
    sis.index = [i.replace("corr_", "").replace("_", " / ") for i in sis.index]
    fig, a = plt.subplots(figsize=(6, 3.2))
    barras = a.barh(sis.index, sis.values, color=c["gris"])
    a.bar_label(barras, fontsize=9)
    a.set_xlabel("Correctivos que tocan el sistema (un correctivo puede tocar varios)")
    a.set_title(f"Correctivos por sistema · {len(corr)} correctivos, 5 camionetas")
    fig.tight_layout()
    guardar(fig, "06_correctivos_por_sistema.png")
    plt.close(fig)


if __name__ == "__main__":
    print("FIGURAS DEL INFORME")
    main()
