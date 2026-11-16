# Predicción de fallas en flotas con ML

Anteproyecto de la Maestría en Inteligencia Analítica de Datos (MIAD), curso Gerencia de Proyectos de Analytics. Esta entrega corresponde a la **Guía 4: preprocesamiento y entendimiento de los datos** del contrato piloto (Contrato D, 5 camionetas).

## Confidencialidad
- `data/raw/` (datos reales) y `data/llaves/` (tabla de códigos) **nunca** se versionan (ver `.gitignore`).
- En el repositorio solo hay datos anonimizados: camionetas `VH-01…VH-05`, conductores `Conductor 01…`, talleres `Taller 01`, órdenes `OS-001…`. No hay coordenadas, calles, municipios, documentos del vehículo ni identificadores internos.
- `src/anonimizar.py` bloquea la exportación si encuentra una placa, un nombre, una cédula o un nombre real de empresa, campo o municipio.

## Cómo reproducir

```bash
pip install -r requirements.txt
python src/anonimizar.py        # solo quien tenga data/raw/ (enlace con la Compañía)
python src/preprocesamiento.py  # limpieza, tabla diaria, ciclos y técnicas de entendimiento
python src/figuras.py           # figuras del informe
```

Los notebooks muestran el análisis paso a paso: `notebooks/02_eda.ipynb` (EDA de las tres fuentes) y `notebooks/03_preprocesamiento.ipynb` (resultados del preprocesamiento).

## Fuentes (Contrato D)
| Fuente | Archivo anonimizado | Periodo |
|---|---|---|
| Consolidado de hojas de vida (Vehículos, Mantenimientos, Plan) | `Consolidado_Hojas_de_Vida_anonimizado.xlsx` | feb-2024 a sep-2026 |
| Eventos de la plataforma GPS | `Detailed Event Report_anonimizado.csv` | sep-2025 a nov-2026 |
| Viajes de la plataforma GPS | `Daily Movement Report_anonimizado.csv` | may-2026 a oct-2026 |

## Anexos técnicos (docs/)
| Archivo | Contenido |
|---|---|
| `log_anonimizacion.csv` | Qué se hizo con cada columna de cada fuente (sin valores reales) |
| `reglas_limpieza.csv` | **Reglas de limpieza**: fuente, campo, causa (a–d de la guía), problema, tratamiento, justificación y registros afectados. Son las reglas que se aplican igual a los datos nuevos |
| `log_preprocesamiento.csv` | Resultado de cada paso, con conteos |
| `factor_odometro.csv` | Sesgo del odómetro GPS frente al km del taller y factor de corrección por camioneta |
| `segmentos_camioneta.csv`, `uso_vs_bateria.csv` | Perfil de uso y de batería por camioneta |
| `importancia_variables.csv`, `correlaciones_altas.csv`, `pca_telemetria.csv` | Técnicas de entendimiento (exploratorias) |
| `figuras/` | Figuras del informe |
