"""
Pre-procesamiento de datos (Guía 4, Partes II y III).

Entrada : data/anonimizados/  (salida de src/anonimizar.py; solo lectura)
          - Consolidado_Hojas_de_Vida_anonimizado.xlsx (Vehículos, Mantenimientos, Plan)
          - Detailed Event Report_anonimizado.csv      (eventos de la plataforma GPS)
          - Daily Movement Report_anonimizado.csv      (viajes de la plataforma GPS)
Salidas : data/procesados/  (tablas fila a fila; ignoradas por Git)
          docs/log_preprocesamiento.csv   (qué pasó en cada paso, con conteos)
          docs/reglas_limpieza.csv        (regla por campo: problema -> tratamiento -> justificación;
                                           es la regla que se aplica igual a los datos nuevos)
          docs/*.csv                      (resultados agregados de las técnicas de entendimiento)

Principios:
- Las tres fuentes usan el MISMO código de camioneta (VH-01...); no hay equivalencias inferidas.
- Ninguna fila se elimina salvo duplicados exactos; los problemas se marcan con banderas.
- Toda variable de la tabla diaria usa solo información disponible hasta ese día
  (sin fuga de información); el objetivo (próximo preventivo) se guarda aparte.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
IN_DIR = BASE / "data" / "anonimizados"
OUT_DIR = BASE / "data" / "procesados"
DOCS_DIR = BASE / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

PAT_PLACA_REAL = r"(?<![A-Za-z0-9])(?!FAC)[A-Za-z]{3}-?\d{3}(?![A-Za-z0-9])"
LAT_RANGO, LON_RANGO = (-5, 14), (-80, -66)  # límites aproximados de Colombia
MAX_HUECO_ODO_DIAS = 7                        # huecos de odómetro que se interpolan
MAX_KM_DIA = 600                              # km/día físicamente posible en la operación
UMBRAL_HUECO_ODO_KM = 1.0                     # salto de odómetro entre viajes que indica viajes no reportados
VIAJE_CORTO_KM = 2.0                          # viaje corto (poca carga de la batería)
EVENTO_VOLTAJE_PROMEDIO = "P.D.C Voltaje Bateria Promedio"
LOG = []
REGLAS = []

# Columnas que no aportan información. Se eliminan solo si, al ejecutar, siguen
# vacías o constantes (la regla se verifica con los datos, no se asume).
COLS_ELIMINAR_EVENTOS = [
    "Asset2ReportGroup", "FleetNumber", "RoadSpeedLimit", "EventLocation",
    "F_StartRegion", "F_EndRegion", "TripID", "FuelUsed",
    "DistanceCalculatedFromOdo", "EndOdo", "TotalOccurs", "TimeZone",
    "AssetSiteName", "ReportGroup", "MeasurementUnits",
]
COLS_ELIMINAR_VIAJES = [
    "FleetNumber", "TripClassificationComment", "PassengerList", "BlankJourneyType",
    "TripType", "NrOfVisits", "TimeZone1", "TimeZone2", "TimeZone", "SiteName",
]

REGLAS_SISTEMA = {
    "frenos": r"fren|pastill|disco|balata",
    "suspension_direccion": r"suspens|amortig|rotul|direcci|terminal|buje",
    "electrico_bateria": r"bater|alternador|arranque|el[eé]ctric|fusible",
    "transmision": r"caja|embrague|clutch|transmis|diferencial",
    "motor": r"motor|inyector|aceite|turbo|culata",
    "refrigeracion": r"radiador|refriger|termostato|bomba de agua|sobrecalent",
}


def registrar(paso, detalle):
    LOG.append({"paso": paso, "detalle": detalle})
    print(f"[{paso}] {detalle}")


def regla(fuente, campo, causa, problema, tratamiento, justificacion, afectados):
    """Regla de limpieza documentada (Guía 4, Parte II).
    causa: a) fuera de rango | b) inválido/vacío/tipo | c) valor extremo | d) relación que no debería existir
    tratamiento: eliminar columna | eliminar fila | imputar | bandera | corregir | no aplicar"""
    REGLAS.append({"fuente": fuente, "campo": campo, "causa_guia": causa, "problema": problema,
                   "tratamiento": tratamiento, "justificacion": justificacion,
                   "registros_afectados": afectados})


def _eliminar_si_sin_informacion(df, columnas, fuente):
    """Elimina las columnas listadas que, en estos datos, están vacías o son constantes."""
    quitar, conservar = [], []
    for c in columnas:
        if c not in df.columns:
            continue
        (quitar if df[c].nunique(dropna=True) <= 1 else conservar).append(c)
    if conservar:
        registrar(f"{fuente}_columnas", f"listadas para eliminar pero con información (se conservan): {conservar}")
    regla(fuente, ", ".join(quitar), "b", "Columnas 100 % vacías o con un único valor (p. ej. TripID = '#Error')",
          "eliminar columna", "No aportan información para ningún objetivo de análisis", len(quitar))
    return df.drop(columns=quitar)


def _slug(texto):
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"\W+", "_", t.lower()).strip("_")


def _a_numero_con_coma_miles(s):
    """Texto tipo '12,326.69' (coma de miles, punto decimal) -> número."""
    return pd.to_numeric(s.astype("string").str.replace(",", "", regex=False), errors="coerce")


def a_numero(s):
    """Hojas de vida: si viene como texto, asume formato colombiano (1.234,5)."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    t = s.astype(str).str.replace(r"[^\d,.\-]", "", regex=True)
    t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(t, errors="coerce")


def clasificar_sistema(texto):
    t = str(texto).lower()
    for sistema, patron in REGLAS_SISTEMA.items():
        if re.search(patron, t):
            return sistema
    return "sin_clasificar"


# ---------- EVENTOS ----------
def cargar_eventos():
    ruta = IN_DIR / "Detailed Event Report_anonimizado.csv"
    return pd.read_csv(ruta, encoding="utf-8-sig", dtype=str)


def limpiar_eventos(ev):
    c0 = ev.shape[1]
    ev = _eliminar_si_sin_informacion(ev, COLS_ELIMINAR_EVENTOS, "eventos").copy()
    n0 = len(ev)
    registrar("columnas", f"eventos: {c0} -> {ev.shape[1]} columnas")

    ev = ev.rename(columns={"AssetExtra": "placa", "AssetName2": "conductor",
                            "EventDescription": "evento", "EventType": "tipo_evento"})
    ev = ev.drop_duplicates(subset="EventKey")
    registrar("duplicados", f"{n0} -> {len(ev)} filas (EventKey único)")

    # Fechas y duración
    fmt = "%d/%m/%Y %H:%M:%S"
    ev["inicio"] = pd.to_datetime(ev["EventStartDate"] + " " + ev["EventStartTime"], format=fmt, errors="coerce")
    ev["fin"] = pd.to_datetime(ev["EventEndDate"] + " " + ev["EventEndTime"], format=fmt, errors="coerce")
    ev["duracion_s"] = pd.to_timedelta(ev["TotalDuration"], errors="coerce").dt.total_seconds()
    p999 = ev.groupby("evento")["duracion_s"].transform(lambda s: s.quantile(0.999))
    ev["duracion_atipica"] = ev["duracion_s"] > p999
    registrar("fechas", f"inicio inválido: {int(ev['inicio'].isna().sum())} | fin < inicio: {int((ev['fin'] < ev['inicio']).sum())}")
    registrar("duracion", f"> 1 h: {int((ev['duracion_s'] > 3600).sum())} | > 24 h: {int((ev['duracion_s'] > 86400).sum())} | atípicas (p99.9 por evento): {int(ev['duracion_atipica'].sum())}")

    regla("eventos", "TotalDuration", "c", f"Duraciones extremas (> 24 h: {int((ev['duracion_s'] > 86400).sum())}), casi todas en pérdidas de cobertura",
          "bandera (duracion_atipica, p99,9 por tipo de evento)",
          "Son reales (la unidad estuvo sin señal); eliminarlas borraría los huecos de monitoreo, que se tratan como dato faltante", int(ev["duracion_atipica"].sum()))

    # Odómetro: 0.00 = no medido en ese tipo de evento (no se imputa por evento)
    ev["odo"] = _a_numero_con_coma_miles(ev["StartOdo"]).replace(0, np.nan)
    o = ev.loc[ev["odo"].notna()].sort_values(["placa", "inicio"])
    retro = int((o.groupby("placa")["odo"].diff() < 0).sum())
    registrar("odometro", f"lecturas reales: {len(o)} ({len(o) / len(ev):.1%}) | retrocesos: {retro}")
    regla("eventos", "StartOdo", "b", "Texto con coma de miles; 0,00 en el 78 % de los eventos (no se mide en ese tipo de evento)",
          "corregir formato; 0 -> faltante", "El 0 no es una lectura: el odómetro solo se reporta en pérdidas de cobertura y desconexiones",
          int(ev["odo"].isna().sum()))

    # Voltaje (mV): el promedio periódico y las alertas son mediciones DISTINTAS.
    # Las alertas solo se disparan por debajo de un umbral: mezclarlas sesga el voltaje hacia abajo.
    valor = _a_numero_con_coma_miles(ev["EventValue"])
    es_prom = ev["evento"].eq(EVENTO_VOLTAJE_PROMEDIO)
    ev["voltaje_mv"] = valor.where(es_prom)                 # solo el promedio periódico
    ev["voltaje_alerta_mv"] = valor.where(~es_prom)         # valor reportado en las alertas
    ev["voltaje_cero"] = es_prom & valor.eq(0)
    ev.loc[ev["voltaje_cero"], "voltaje_mv"] = np.nan
    fuera = es_prom & (ev["voltaje_mv"] > 16000)
    ev.loc[fuera, "voltaje_mv"] = np.nan
    registrar("voltaje", f"promedios: {int(es_prom.sum())} | en 0 mV (-> faltante): {int(ev['voltaje_cero'].sum())} | "
                         f"> 16 V (-> faltante): {int(fuera.sum())} | alertas con valor: {int(ev['voltaje_alerta_mv'].notna().sum())}")
    regla("eventos", "EventValue (voltaje promedio)", "a", "Lecturas en 0 mV con el vehículo reportando eventos",
          "faltante (no se imputa)", "Un 0 V no es físicamente posible con la unidad encendida: es pérdida de alimentación del equipo; "
          "imputarlo inventaría una medición de la señal que se quiere vigilar", int(ev["voltaje_cero"].sum()))
    regla("eventos", "EventValue (alertas)", "d", "Las alertas (<10 V, <9,5 V) traen valor de voltaje, pero solo existen cuando el voltaje cae",
          "separar: voltaje_mv (promedio) y voltaje_alerta_mv", "Mezclarlas con los promedios sesga la mediana del voltaje hacia abajo; "
          "las alertas se usan como conteo diario", int(ev["voltaje_alerta_mv"].notna().sum()))
    regla("eventos", "EventValue (voltaje promedio)", "c", "Caídas de voltaje profundas (hasta ~3 V en promedio diario)",
          "conservar", "Son la señal de deterioro de la batería que busca OA3; solo se descartan valores físicamente imposibles", 0)

    # Región (la anonimización eliminó coordenadas, calles y municipios; queda la región)
    for col, nuevo in (("F_StartTown", "region_ini"), ("F_EndTown", "region_fin")):
        if col in ev:
            ev[nuevo] = ev[col].replace("Sin dato", np.nan)
    ev["sin_gps_inicio"] = ev["region_ini"].isna() if "region_ini" in ev else False
    registrar("ubicacion", f"eventos sin posición (región 'Sin dato'): {int(ev['sin_gps_inicio'].sum())} ({ev['sin_gps_inicio'].mean():.1%})")
    regla("eventos", "Ubicación", "b", "Sin posición GPS en ~19 % de los eventos",
          "no aplicar imputación", "Ningún objetivo de análisis usa la ubicación; imputar coordenadas no aporta al problema", int(ev["sin_gps_inicio"].sum()))

    # Conductor sin identificar
    ev["conductor_sin_identificar"] = ev["conductor"].eq("Conductor sin identificar")
    regla("eventos", "AssetName2 (conductor)", "b", "Conductor sin identificar en ~22 % de los eventos",
          "bandera (conductor_sin_identificar)", "Los conductores rotan entre camionetas (11 a 18 por vehículo); no se usa para atribuir el estado de la batería", int(ev["conductor_sin_identificar"].sum()))

    # Se descartan las columnas de texto ya convertidas
    ev = ev.drop(columns=["EventStartDate", "EventStartTime", "EventEndDate", "EventEndTime",
                          "TotalDuration", "StartOdo", "EventValue", "F_StartTown", "F_EndTown"], errors="ignore")

    # Control de privacidad: ninguna placa real debe sobrevivir
    for c in ev.columns:
        if pd.api.types.is_object_dtype(ev[c]) or pd.api.types.is_string_dtype(ev[c]):
            n = ev[c].astype(str).str.contains(PAT_PLACA_REAL, regex=True).sum()
            if n:
                raise ValueError(f"Posible placa real en la columna {c}: {n} filas")
    registrar("final_eventos", f"{ev.shape[0]} filas x {ev.shape[1]} columnas")
    return ev


def _interpolar_huecos_cortos(s, max_dias=MAX_HUECO_ODO_DIAS):
    """Interpola linealmente solo los huecos de hasta max_dias; los largos quedan en NaN."""
    interp = s.interpolate(limit_area="inside")
    es_nan = s.isna()
    grupo = (es_nan != es_nan.shift()).cumsum()
    largo = es_nan.groupby(grupo).transform("sum")
    interp[es_nan & (largo > max_dias)] = np.nan
    return interp


def agregar_eventos_dia(ev):
    """Camioneta x día (calendario completo entre su primer y último evento)."""
    d = ev.dropna(subset=["inicio", "placa"]).copy()
    d["fecha"] = d["inicio"].dt.normalize()
    idx = ["placa", "fecha"]

    conteo = d.pivot_table(index=idx, columns="evento", values="EventKey", aggfunc="count", fill_value=0)
    conteo.columns = ["n_" + _slug(c) for c in conteo.columns]

    volt = (d[d["voltaje_mv"] > 0].groupby(idx)["voltaje_mv"]
              .agg(voltaje_min="min", voltaje_medio="mean", n_lecturas_voltaje="count"))
    ralenti = (d[d["evento"].str.startswith("EC Ralenti")].groupby(idx)["duracion_s"]
                 .sum().rename("duracion_ralenti_s"))
    extra = d.groupby(idx).agg(n_eventos=("EventKey", "size"),
                               n_voltaje_cero=("voltaje_cero", "sum"),
                               n_sin_gps=("sin_gps_inicio", "sum"),
                               odo_lecturas=("odo", "count"),
                               odo_dia=("odo", "max"))
    tabla = conteo.join(extra).join(volt).join(ralenti).reset_index()

    cols_cero = [c for c in tabla.columns if c.startswith("n_")] + ["odo_lecturas", "duracion_ralenti_s"]
    partes = []
    for placa, g in tabla.groupby("placa"):
        g = g.drop(columns="placa").set_index("fecha").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        g["dia_con_eventos"] = g["n_eventos"].notna()
        g[cols_cero] = g[cols_cero].fillna(0)
        g["odo_interp"] = _interpolar_huecos_cortos(g["odo_dia"])
        g["odo_interpolado"] = g["odo_dia"].isna() & g["odo_interp"].notna()
        g["km_dia"] = g["odo_interp"].diff()
        g.insert(0, "placa", placa)
        partes.append(g.rename_axis("fecha").reset_index())
    diario = pd.concat(partes, ignore_index=True)

    registrar("diario", f"{len(diario)} camioneta-días | sin eventos: {int((~diario['dia_con_eventos']).sum())}")
    registrar("km_dia", f"con dato: {int(diario['km_dia'].notna().sum())} | interpolados (huecos <= {MAX_HUECO_ODO_DIAS} d): {int(diario['odo_interpolado'].sum())}")
    for placa, g in diario.groupby("placa"):
        registrar("odo_cobertura", f"{placa}: km_dia con dato en {g['km_dia'].notna().mean():.0%} de {len(g)} días")
    return diario


# ---------- VIAJES (Daily Movement Report) ----------
def cargar_viajes():
    ruta = IN_DIR / "Daily Movement Report_anonimizado.csv"
    return pd.read_csv(ruta, encoding="utf-8-sig", dtype=str) if ruta.exists() else None


def limpiar_viajes(vj):
    c0, n0 = vj.shape[1], len(vj)
    vj = _eliminar_si_sin_informacion(vj, COLS_ELIMINAR_VIAJES, "viajes").copy()
    vj = vj.rename(columns={"AssetExtra": "placa", "AssetName2": "conductor"})
    fmt = "%d/%m/%Y %H:%M:%S"
    vj["salida"] = pd.to_datetime(vj["DepartureDate"] + " " + vj["DepartureTime"], format=fmt, errors="coerce")
    vj["llegada"] = pd.to_datetime(vj["ArrivalDate"] + " " + vj["ArrivalTime"], format=fmt, errors="coerce")
    for c in ("Distance", "StartOdoMeter", "EndOdoMeter", "MaxSpeed", "AvgSpeed", "TotalFuelUsedMeasured", "CalcFuelConsumption"):
        vj[c] = _a_numero_con_coma_miles(vj[c])
    for c in ("DrivingTime", "StandingTime", "Duration"):
        vj[c + "_s"] = pd.to_timedelta(vj[c], errors="coerce").dt.total_seconds()
    regla("viajes", "Distance, odómetros, tiempos", "b", "Números como texto con coma de miles; fechas y horas en columnas separadas",
          "corregir formato", "Necesario para sumar km y construir la serie diaria", len(vj))

    # Duplicados: misma camioneta, hora de salida y odómetro (aparecen al cierre del reporte)
    dup = vj.duplicated(["placa", "salida", "StartOdoMeter"])
    vj = vj[~dup].sort_values(["placa", "salida"]).reset_index(drop=True)
    regla("viajes", "Viaje", "b", "Viajes repetidos (misma camioneta, hora y odómetro)", "eliminar fila",
          "Duplicado exacto: contarlo dos veces infla los km del día", int(dup.sum()))

    # Consistencia interna: distancia = diferencia de odómetro
    dif = (vj["EndOdoMeter"] - vj["StartOdoMeter"] - vj["Distance"]).abs()
    registrar("viajes_consistencia", f"viajes con |distancia - (odo fin - odo ini)| > 0,1 km: {int((dif > 0.1).sum())} de {len(vj)}")

    # Huecos: salto de odómetro entre el fin de un viaje y el inicio del siguiente = viajes no reportados
    vj["hueco_odo_km"] = vj["StartOdoMeter"] - vj.groupby("placa")["EndOdoMeter"].shift()
    huecos = vj["hueco_odo_km"] > UMBRAL_HUECO_ODO_KM
    retro = vj["hueco_odo_km"] < -UMBRAL_HUECO_ODO_KM
    registrar("viajes_huecos", f"saltos de odómetro > {UMBRAL_HUECO_ODO_KM} km entre viajes: {int(huecos.sum())} "
                               f"(suman {vj.loc[huecos, 'hueco_odo_km'].sum():,.0f} km) | retrocesos: {int(retro.sum())}")
    regla("viajes", "StartOdoMeter vs. viaje anterior", "d", "Salto de odómetro entre viajes consecutivos (viajes no reportados)",
          "bandera; los días del hueco quedan sin dato (no se imputan como 0 km ni se cargan al día de reinicio)",
          "Distinguir 'sin dato' de 'sin uso': el hueco de 4.360 km de VH-05 coincide con 51 días sin viajes", int(huecos.sum()))

    # Consumo: valores absurdos en viajes muy cortos
    absurdo = vj["Distance"] < 1
    regla("viajes", "CalcFuelConsumption", "c", f"Consumo calculado de hasta {vj['CalcFuelConsumption'].max():,.0f} en viajes de pocos metros",
          "no usar por viaje; recalcular por día excluyendo viajes < 1 km", "El cociente combustible/distancia explota cuando la distancia tiende a 0",
          int(absurdo.sum()))
    registrar("viajes", f"{n0} -> {len(vj)} filas | {c0} -> {vj.shape[1]} columnas | periodo {vj['salida'].min():%Y-%m-%d} a {vj['salida'].max():%Y-%m-%d}")
    return vj


def agregar_viajes_dia(vj):
    """Camioneta x día desde los viajes. Un día sin viajes es 'sin uso' (0 km) si el odómetro
    no cambió entre el viaje anterior y el siguiente; si cambió, es 'sin dato'."""
    vj = vj.copy()
    vj["fecha"] = vj["salida"].dt.normalize()
    vj["viaje_corto"] = vj["Distance"] < VIAJE_CORTO_KM
    largos = vj[vj["Distance"] >= 1]
    g = vj.groupby(["placa", "fecha"])
    dia = g.agg(km_dia_viajes=("Distance", "sum"), n_viajes=("Distance", "size"),
                n_viajes_cortos=("viaje_corto", "sum"), horas_conduccion=("DrivingTime_s", "sum"),
                vel_max=("MaxSpeed", "max"), odo_ini_viajes=("StartOdoMeter", "min"),
                odo_fin_viajes=("EndOdoMeter", "max"), hueco_km=("hueco_odo_km", lambda s: s[s > UMBRAL_HUECO_ODO_KM].sum()))
    dia["horas_conduccion"] /= 3600
    comb = largos.groupby(["placa", "fecha"]).agg(comb=("TotalFuelUsedMeasured", "sum"), dist=("Distance", "sum"))
    dia["litros_100km"] = (comb["comb"] / comb["dist"] * 100).where(comb["dist"] > 5)
    partes = []
    ini, fin = vj["fecha"].min(), vj["fecha"].max()
    for placa, gp in dia.reset_index().groupby("placa"):
        gp = gp.set_index("fecha").reindex(pd.date_range(ini, fin, freq="D"))
        gp["placa"] = placa
        gp["dia_con_viajes"] = gp["n_viajes"].notna()
        # ¿El día sin viajes está dentro de un hueco de odómetro?
        odo_prev = gp["odo_fin_viajes"].ffill()
        odo_sig = gp["odo_ini_viajes"].bfill()
        sin_uso = (~gp["dia_con_viajes"] & ((odo_sig - odo_prev).abs() <= UMBRAL_HUECO_ODO_KM)).fillna(False).astype(bool)
        gp.loc[sin_uso, ["km_dia_viajes", "n_viajes", "n_viajes_cortos", "horas_conduccion"]] = 0
        gp["estado_dia"] = np.select([gp["dia_con_viajes"].to_numpy(bool), sin_uso.to_numpy(bool)], ["con viajes", "sin uso"], "sin dato")
        # El día en que se reanudan los datos después de un hueco no recibe los km del hueco
        partes.append(gp.rename_axis("fecha").reset_index())
    out = pd.concat(partes, ignore_index=True)
    resumen = out.groupby("placa")["estado_dia"].value_counts().unstack(fill_value=0)
    registrar("viajes_dia", "días por estado y camioneta: " + str(resumen.to_dict(orient="index")))
    return out


def factor_odometro(viajes_dia, eventos_dia, m):
    """Factor km_taller / odómetro GPS por camioneta, con los días de preventivo.
    El sesgo es proporcional (la telemetría pierde km cuando no tiene señal), por eso es un
    factor y no un desfase fijo."""
    prev = m[~m["es_correctivo"]][["placa", "fecha", "km"]].copy()
    prev["fecha"] = prev["fecha"].astype("datetime64[ns]")
    filas = []
    for fuente, tabla, col in (("viajes", viajes_dia, "odo_ini_viajes"), ("eventos", eventos_dia, "odo_interp")):
        if tabla is None or col not in tabla:
            continue
        t = tabla[["placa", "fecha", col]].dropna()
        t["fecha"] = t["fecha"].astype("datetime64[ns]")
        x = prev.merge(t, on=["placa", "fecha"])
        x["factor"] = x["km"] / x[col]
        x["fuente"] = fuente
        filas.append(x[["placa", "fecha", "km", col, "factor", "fuente"]].rename(columns={col: "odo_gps"}))
    cruces = pd.concat(filas, ignore_index=True)
    cruces["sesgo_pct"] = (cruces["odo_gps"] / cruces["km"] - 1) * 100
    factor = cruces.groupby("placa")["factor"].median()
    resumen = cruces.groupby(["placa", "fuente"])["sesgo_pct"].agg(["median", "count"]).round(2)
    registrar("factor_odometro", "sesgo GPS vs. taller (% mediano, n): " + str(resumen.to_dict(orient="index")))
    regla("eventos/viajes", "Odómetro GPS", "d", f"El odómetro GPS queda por debajo del km del taller "
          f"({cruces['sesgo_pct'].min():.1f} % a {cruces['sesgo_pct'].max():.1f} %); riesgo 1.2.1 de la Guía 3",
          "corregir: factor por camioneta (mediana de km_taller / odo_gps en días de preventivo)",
          "El sesgo es estable dentro de cada camioneta y crece con los km (pérdida de señal), por eso un factor y no un desfase fijo",
          len(cruces))
    return factor, cruces


# ---------- HOJAS DE VIDA ----------
# Desde la nueva anonimización, las tres fuentes usan el mismo código de camioneta.
# Se conserva la columna placa_eventos (= placa) por compatibilidad con el resto del código.

# Sistemas afectados en los correctivos (texto libre; una descripción puede tocar varios)
SISTEMAS_CORRECTIVO = {
    "frenos": r"pastill|disco|zapata|fren",
    "suspension_direccion": r"buje|estabiliz|amortigu|torsi|suspens|rotul|direcci|terminal",
    "bateria_electrico": r"bater|alarma|alternador|arranque|fusible",
    "caja_transmision": r"caja|embrague|clutch|transmis|diferencial",
    "correas_accesorios": r"correa|polea|accesorios",
    "motor_admision": r"manguera|admisi|motor|inyector|turbo",
    "carroceria_otros": r"plumilla",
}


def cargar_hojas():
    return pd.read_excel(IN_DIR / "Consolidado_Hojas_de_Vida_anonimizado.xlsx", sheet_name=None)


def _km_nominal(desc):
    """'MTTO PREVENTIVO 5.000KM' / '20.000 KM' -> 5000 / 20000. Tolera el typo '65.0.000KM' -> 65000."""
    mt = re.search(r"(\d[\d.]*\d|\d)\s*km", str(desc), flags=re.I)
    if not mt:
        return np.nan
    tok = mt.group(1)
    partes = tok.split(".")
    if len(partes) == 1:
        return int(tok)
    if all(len(x) == 3 for x in partes[1:]):
        return int("".join(partes))
    return int(partes[0]) * 1000


def limpiar_vehiculos(hoja, fecha_corte):
    vacias = [c for c in hoja.columns if hoja[c].isna().all()]
    consts = [c for c in hoja.columns if c not in vacias and c != "Placa" and hoja[c].nunique(dropna=False) <= 1]
    registrar("vehiculos_columnas", f"{hoja.shape[1]} originales | vacías: {len(vacias)} | constantes: {len(consts)}")
    v = hoja[["Placa", "Carrocería", "Color", "Capacidad (kg)", "Fecha expedición de placa"]].rename(columns={
        "Placa": "placa", "Carrocería": "carroceria", "Color": "color",
        "Capacidad (kg)": "capacidad_kg", "Fecha expedición de placa": "fecha_placa"})
    typos = int(v["carroceria"].astype(str).str.contains("CABIAN").sum())
    v["carroceria"] = v["carroceria"].astype(str).str.strip().str.upper().str.replace("CABIAN", "CABINA", regex=False)
    registrar("vehiculos_typos", f"carrocería con 'CABIAN' corregido a 'CABINA': {typos}")
    regla("vehiculos", "Carrocería", "b", "Error de digitación ('CABIAN') y un valor truncado ('DOBLE CABINA CON')",
          "corregir el error de digitación; el truncado se conserva", "No hay forma de saber cómo terminaba el texto truncado; no lo usa ningún objetivo", typos + 1)
    regla("vehiculos", f"{len(vacias) + len(consts)} columnas", "b", "Columnas vacías (tanque, extintor, facturas) o constantes (marca, línea, modelo, equipamiento)",
          "eliminar columna", "Las 5 camionetas son Nissan Frontier 2024 con el mismo equipamiento: no diferencian camionetas", len(vacias) + len(consts))
    v["fecha_placa"] = pd.to_datetime(v["fecha_placa"], errors="coerce")
    v["antiguedad_dias"] = (fecha_corte - v["fecha_placa"]).dt.days
    registrar("vehiculos", f"{len(v)} filas x {v.shape[1]} columnas (antigüedad al {fecha_corte.date()})")
    return v


def limpiar_mantenimientos(mant):
    m = mant.rename(columns={
        "Placa": "placa", "Fecha": "fecha", "Descripción": "descripcion", "Tipo": "tipo",
        "Kilometraje acumulado": "km", "Orden de servicio": "orden",
        "Costo (COP)": "costo", "Prestador del servicio": "prestador",
        "Orden de servicio formato válido": "orden_formato_valido"}).copy()
    n = len(m)
    m["fecha"] = pd.to_datetime(m["fecha"], dayfirst=True, errors="coerce")
    m["km"], m["costo"] = a_numero(m["km"]), a_numero(m["costo"])
    m["tipo"] = m["tipo"].astype(str).str.strip().str.lower()
    m = m.drop_duplicates().dropna(subset=["placa", "fecha"])
    registrar("mantenimientos", f"{n} -> {len(m)} filas | nulos: {int(m.isna().sum().sum())}")

    m["es_correctivo"] = m["tipo"].str.contains("correct")
    m = m.sort_values(["placa", "fecha", "es_correctivo"]).reset_index(drop=True)

    # Visita = placa + fecha (un preventivo y un correctivo el mismo día comparten orden y km)
    g = m.groupby(["placa", "fecha"])
    m["n_filas_visita"] = g["orden"].transform("size")
    m["visita_con_correctivo"] = g["es_correctivo"].transform("any")
    m["costo_visita"] = g["costo"].transform("sum")
    registrar("visitas", f"{g.ngroups} visitas | con preventivo + correctivo el mismo día: "
                         f"{int(((m['n_filas_visita'] > 1) & m['visita_con_correctivo']).sum() / 2)}")

    # (d) Relaciones que no deberían existir
    prev_km = m[~m["es_correctivo"]][["placa", "fecha", "km"]].rename(columns={"fecha": "fecha_prev", "km": "km_prev"})
    x = m[m["es_correctivo"]].reset_index().merge(prev_km, on="placa")
    x = x[(x["km"] == x["km_prev"]) & (x["fecha"] > x["fecha_prev"])]
    m["km_copiado"] = m.index.isin(x["index"])
    regla("mantenimientos", "Kilometraje acumulado", "d", "Correctivos registrados días después de un preventivo con el MISMO km del preventivo",
          "bandera (km_copiado); el km de esos correctivos no se usa para ubicar la falla en el ciclo",
          "El km fue copiado del preventivo: la camioneta siguió rodando entre ambas fechas", int(m["km_copiado"].sum()))
    dup_costo = m.duplicated(["placa", "fecha", "costo"], keep=False) & (m.groupby(["placa", "fecha"])["tipo"].transform("nunique") > 1)
    m["costo_duplicado"] = dup_costo
    regla("mantenimientos", "Costo (COP)", "d", "Preventivo y correctivo de la misma visita con exactamente el mismo costo",
          "bandera (costo_duplicado); se excluyen del análisis de costos por tipo",
          "Es muy improbable que dos trabajos distintos cuesten lo mismo al peso: probable copia del valor de la factura", int(dup_costo.sum()))
    os_placas = m.groupby("orden")["placa"].nunique()
    m["orden_en_varias_placas"] = m["orden"].map(os_placas) > 1
    regla("mantenimientos", "Orden de servicio", "d", "La misma orden de servicio aparece en dos camionetas distintas",
          "bandera (orden_en_varias_placas)", "Una orden corresponde a una camioneta; se valida con Abastecimiento y Logística", int(m["orden_en_varias_placas"].sum()))
    if "orden_formato_valido" in m:
        malos = int((~m["orden_formato_valido"].astype(bool)).sum())
        regla("mantenimientos", "Orden de servicio", "a", "Números de orden fuera del patrón de 8 dígitos (9 dígitos o prefijo 'R')",
              "bandera (orden_formato_valido, calculada antes de anonimizar)", "Errores de digitación; la orden no se usa como variable, solo como llave de visita", malos)
    regla("mantenimientos", "Prestador del servicio", "b", "El mismo taller escrito de 3 formas distintas",
          "corregir (normalizado en la anonimización -> Taller 01)", "Sin normalizar, el análisis por taller contaría tres prestadores", int(len(m)))

    # Preventivos: km nominal (del texto) e intervalos entre preventivos
    m["km_nominal"] = np.where(~m["es_correctivo"], m["descripcion"].map(_km_nominal), np.nan)
    sin_nominal = int((~m["es_correctivo"] & m["km_nominal"].isna()).sum())
    typo_nom = int(m["descripcion"].astype(str).str.contains(r"\d\.\d\.\d", regex=True).sum())
    registrar("km_nominal_typo", f"descripciones con typo de miles corregidas: {typo_nom}")
    prev = m[~m["es_correctivo"]]
    m["km_desde_prev_anterior"] = prev.groupby("placa")["km"].diff()
    m["dias_desde_prev_anterior"] = prev.groupby("placa")["fecha"].diff().dt.days
    m["km_decrece"] = m.groupby("placa")["km"].diff() < 0
    registrar("preventivos", f"{len(prev)} | km nominal no extraído: {sin_nominal} | "
                             f"km entre preventivos (mediana): {prev.groupby('placa')['km'].diff().median():.0f} | "
                             f"días entre preventivos (mediana): {prev.groupby('placa')['fecha'].diff().dt.days.median():.0f}")
    registrar("km", f"decrecientes: {int(m['km_decrece'].sum())}")
    regla("mantenimientos", "Descripción (preventivo)", "b", "Km nominal con errores de digitación ('65.0.000KM', con y sin espacio)",
          "corregir (extracción tolerante del km nominal)", "El km nominal define el servicio del plan y el ciclo", typo_nom)
    salto = prev.groupby("placa")["km"].diff()
    regla("mantenimientos", "Intervalo entre preventivos", "c", f"Intervalos extremos: {int((salto > 9000).sum())} > 9.000 km (servicio omitido) y "
          f"{int((salto < 3500).sum())} < 3.500 km (servicio adelantado)", "conservar con bandera en el ciclo",
          "Son reales: un preventivo omitido es justamente lo que el pronóstico debe ayudar a evitar", int(((salto > 9000) | (salto < 3500)).sum()))

    # Correctivos: sistemas afectados (varias banderas por descripción)
    corr = m["es_correctivo"]
    for sistema, patron in SISTEMAS_CORRECTIVO.items():
        m[f"corr_{sistema}"] = corr & m["descripcion"].str.lower().str.contains(patron, regex=True)
    cols_sis = [f"corr_{s}" for s in SISTEMAS_CORRECTIVO]
    sin_sis = int((corr & ~m[cols_sis].any(axis=1)).sum())
    registrar("correctivos", f"{int(corr.sum())} | sin sistema clasificado: {sin_sis} | "
                             + str({s: int(m[f'corr_{s}'].sum()) for s in SISTEMAS_CORRECTIVO}))
    multi = int((m[cols_sis].sum(axis=1) > 1).sum())
    regla("mantenimientos", "Descripción (correctivo)", "b", f"Texto libre sin catálogo de sistemas; {multi} correctivos tocan más de un sistema",
          "clasificar por palabras clave con etiquetas múltiples (corr_<sistema>)", "Es la base de la variable objetivo por sistema (OA1/OA4); "
          "queda pendiente validarla con el Champion (F1 >= 0,90)", int(corr.sum()))
    return m


def limpiar_plan(plan, m):
    p = plan.rename(columns={
        "Placa": "placa", "Año del plan": "anio_plan", "Rutina": "rutina",
        "Servicio del plan (km)": "servicio_km", "Km programado": "km_prog_ref",
        "Fecha programada": "fecha_prog_ref", "Km ejecutado": "km_ejec",
        "Fecha ejecutada": "fecha_ejec", "Observación": "observacion"}).drop(
        columns=["Texto original programado", "Texto original ejecutado"], errors="ignore").copy()
    for c in ("fecha_prog_ref", "fecha_ejec"):
        p[c] = pd.to_datetime(p[c], dayfirst=True, errors="coerce")
    for c in ("km_prog_ref", "km_ejec", "servicio_km"):
        p[c] = a_numero(p[c])
    # km_prog_ref y fecha_prog_ref NO son confiables (estimaciones); la referencia es servicio_km
    p["fecha_prog_invalida"] = p["fecha_prog_ref"].isna() & p["observacion"].astype(str).str.contains("inválida", case=False)
    p["desv_km_vs_nominal"] = p["km_ejec"] - p["servicio_km"]
    p["ejecutado"] = p["km_ejec"].notna() | p["fecha_ejec"].notna()

    # Estado de cada servicio del plan, conciliado con la hoja de mantenimientos:
    #  ejecutado | conflicto_plan_vs_mant (plan dice no ejecutado, mantenimientos sí) | omitido | pendiente
    prev = m[~m["es_correctivo"]]
    nominales = prev.groupby("placa")["km_nominal"].apply(set).to_dict()
    p["estado"] = "ejecutado"
    for idx in p.index[~p["ejecutado"]]:
        r = p.loc[idx]
        posterior = p[(p["placa"] == r["placa"]) & p["ejecutado"] & (p["servicio_km"] > r["servicio_km"])]
        if r["servicio_km"] in nominales.get(r["placa"], set()):
            p.loc[idx, "estado"] = "conflicto_plan_vs_mant"
        elif len(posterior):
            p.loc[idx, "estado"] = "omitido"
        else:
            p.loc[idx, "estado"] = "pendiente"
    registrar("plan", f"filas: {len(p)} | " + str(p["estado"].value_counts().to_dict()))

    # Conciliación de lo ejecutado: ¿hay un preventivo el mismo día en mantenimientos? ¿con el mismo km?
    ej = p[p["ejecutado"]].merge(prev[["placa", "fecha", "km", "km_nominal"]],
                                 left_on=["placa", "fecha_ejec"], right_on=["placa", "fecha"], how="left")
    registrar("conciliacion", f"plan ejecutado con preventivo el mismo día: {ej['fecha'].notna().mean():.1%} | "
                              f"mismo km: {(ej['km_ejec'] == ej['km']).mean():.1%} | "
                              f"mismo servicio nominal: {(ej['servicio_km'] == ej['km_nominal']).mean():.1%}")
    registrar("plan_desv_km", "vs nominal (mediana, p75, max): "
              f"{p['desv_km_vs_nominal'].median():.0f}, {p['desv_km_vs_nominal'].quantile(.75):.0f}, {p['desv_km_vs_nominal'].max():.0f}")
    registrar("plan_fechas", f"fecha programada inválida en origen: {int(p['fecha_prog_invalida'].sum())}")

    # Tipo de discrepancia de cada servicio ejecutado frente a Mantenimientos (preventivo más cercano en km)
    def discrepancia(r):
        g = prev[prev["placa"] == r["placa"]]
        if g.empty or pd.isna(r["km_ejec"]):
            return pd.Series({"tipo_discrepancia": "sin dato", "km_mant": np.nan, "fecha_mant": pd.NaT})
        k = g.iloc[(g["km"] - r["km_ejec"]).abs().argsort()[:1]].iloc[0]
        dkm, ddias = r["km_ejec"] - k["km"], (r["fecha_ejec"] - k["fecha"]).days if pd.notna(r["fecha_ejec"]) else np.nan
        if dkm == 0 and ddias == 0:
            t = "coincide"
        elif dkm == 0 and abs(abs(ddias) - 365) <= 2:
            t = "error de año en la fecha"
        elif dkm == 0:
            t = f"fecha distinta ({ddias:+.0f} d)"
        elif r["km_ejec"] % 1000 == 0 and abs(dkm) < 3000:
            t = "km nominal en lugar del real"
        elif abs(dkm) <= 1000 and sum(a != b for a, b in zip(str(int(r["km_ejec"])), str(int(k["km"])))) == 1:
            t = "error de un dígito en el km"
        else:
            t = "otra"
        return pd.Series({"tipo_discrepancia": t, "km_mant": k["km"], "fecha_mant": k["fecha"]})
    p = p.join(p.apply(discrepancia, axis=1))
    registrar("plan_discrepancias", str(p.loc[p["ejecutado"], "tipo_discrepancia"].value_counts().to_dict()))
    no_coinc = int((p["ejecutado"] & ~p["tipo_discrepancia"].eq("coincide")).sum())
    regla("plan", "Km ejecutado / Fecha ejecutada", "d", "El plan no coincide con Mantenimientos (error de un dígito, año errado, km nominal en lugar del real)",
          "corregir: la hoja Mantenimientos es la fuente de verdad; el plan se usa solo para el estado del servicio",
          "Mantenimientos se alimenta de la orden de servicio; el plan se digita aparte", no_coinc)
    regla("plan", "Fecha programada / Km programado", "d", "La fecha programada es casi siempre POSTERIOR a la ejecutada (mediana ~+138 d)",
          "eliminar columna para el análisis (se usa servicio_km como referencia)", "No es un plan previo: no sirve como objetivo ni como predictor", int(p["fecha_prog_ref"].notna().sum()))
    regla("plan", "Fecha programada", "a", "Fecha inválida en origen (31/09/2025)", "faltante", "No existe esa fecha", int(p["fecha_prog_invalida"].sum()))
    return p


def limpiar_hojas(hv):
    fecha_corte = pd.to_datetime(hv["Mantenimientos"]["Fecha"], errors="coerce").max()
    v = limpiar_vehiculos(hv["Vehículos"], fecha_corte)
    m = limpiar_mantenimientos(hv["Mantenimientos"])
    p = limpiar_plan(hv["Plan de mantenimiento"], m)

    # Equivalencia con las placas de eventos (se conserva la placa original de hojas de vida)
    for df in (v, m, p):
        df["placa_eventos"] = df["placa"]
    m = etiquetar_no_programado(m)

    # Control de privacidad
    for nombre, df in (("vehiculos", v), ("mantenimientos", m), ("plan", p)):
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
                n = df[c].astype(str).str.contains(PAT_PLACA_REAL, regex=True).sum()
                if n:
                    raise ValueError(f"Posible placa real en {nombre}.{c}: {n} filas")
    return m, p, v


def cruce(ev, v, m):
    placas = set(ev["placa"].dropna().unique())
    for nombre, df in (("vehiculos", v), ("mantenimientos", m)):
        comunes = placas & set(df["placa_eventos"].dropna().unique())
        registrar("cruce", f"{nombre} (por equivalencia): {len(comunes)} de {len(placas)} placas de eventos ({len(comunes)/max(len(placas),1):.0%})")



# =====================================================================================
# ETIQUETA DE NO PROGRAMADO, CICLOS, UNIÓN CON EL ÚLTIMO PREVENTIVO Y TÉCNICAS DE ENTENDIMIENTO
# =====================================================================================
UMBRAL_NO_PROG_DIAS = 3        # Guía 2: correctivo a más de 3 días de un preventivo = no programado
SALTO_NOMINAL_KM = 5000        # el plan de mantenimiento es cada 5.000 km
COBERTURA_MIN_CICLO = 0.8      # un ciclo se usa con telemetría si hay eventos en >= 80 % de sus días
USO_ALTO_KM_DIA, USO_BAJO_KM_DIA = 100, 50
CORR_ALTA = 0.9                # umbral de correlación para sugerir descartar una variable
RESULTADOS = {}                # tablas nuevas, accesibles desde el notebook: pp.RESULTADOS["ciclos"]


def etiquetar_no_programado(m):
    """Marca cada correctivo según su distancia en días al preventivo más cercano de la misma placa."""
    m = m.copy()
    prev = m.loc[~m["es_correctivo"], ["placa", "fecha"]]

    def dias(r):
        f = prev.loc[prev["placa"] == r["placa"], "fecha"]
        return (f - r["fecha"]).abs().dt.days.min() if len(f) else np.nan

    c = m["es_correctivo"]
    m["dias_a_preventivo"] = np.nan
    m.loc[c, "dias_a_preventivo"] = m[c].apply(dias, axis=1)
    m["en_visita_preventivo"] = c & (m["dias_a_preventivo"] == 0)
    m["no_programado"] = c & (m["dias_a_preventivo"] > UMBRAL_NO_PROG_DIAS)

    n = int(c.sum())
    sens = {u: f"{(m.loc[c, 'dias_a_preventivo'] > u).mean():.0%}" for u in (0, 1, 3, 5, 7, 15)}
    registrar("no_programado", f"{int(m['no_programado'].sum())} de {n} correctivos a más de {UMBRAL_NO_PROG_DIAS} d de un preventivo "
                               f"({m['no_programado'].sum() / max(n, 1):.0%}); en la misma visita del preventivo: {int(m['en_visita_preventivo'].sum())}")
    registrar("no_programado_sensibilidad", f"% de correctivos a más de X días de un preventivo: {sens}")
    registrar("no_programado_baseline", "Las guías reportan 70 % (270 de 384, flota completa 2025); esta muestra son 5 camionetas y 19 correctivos: no es comparable")
    return m


def _nominal_siguiente(nominal_ult, km_ult):
    """Siguiente servicio nominal: el anterior + 5.000; si ese ya quedó atrás por el km real, el múltiplo de 5.000 siguiente."""
    sig = nominal_ult + SALTO_NOMINAL_KM
    atras = sig <= km_ult
    sig = sig.where(~atras, (np.floor(km_ult / SALTO_NOMINAL_KM) + 1) * SALTO_NOMINAL_KM)
    return sig


def unir_fuentes_diarias(ev_dia, vj_dia):
    """Tabla camioneta x día con las dos fuentes de telemetría.
    km_dia: se toma de los viajes cuando el día tiene dato de viajes (fuente directa);
    si no, de la diferencia del odómetro de los eventos (fuente indirecta)."""
    d = ev_dia.rename(columns={"km_dia": "km_dia_eventos"})
    if vj_dia is not None:
        d = d.merge(vj_dia, on=["placa", "fecha"], how="outer")
        d["dia_con_eventos"] = d["dia_con_eventos"].fillna(False).astype(bool)
        usar_viajes = d["estado_dia"].isin(["con viajes", "sin uso"])
        d["km_dia"] = np.where(usar_viajes, d["km_dia_viajes"], d["km_dia_eventos"])
        d["fuente_km_dia"] = np.select([usar_viajes.to_numpy(bool), d["km_dia_eventos"].notna().to_numpy(bool)], ["viajes", "eventos"], "sin dato")
    else:
        d["km_dia"], d["fuente_km_dia"] = d["km_dia_eventos"], np.where(d["km_dia_eventos"].notna(), "eventos", "sin dato")
    excesivo = d["km_dia"] > MAX_KM_DIA
    d.loc[excesivo, "km_dia"] = np.nan
    regla("tabla diaria", "km_dia", "a", f"km/día > {MAX_KM_DIA} (imposible en la operación del contrato)", "faltante",
          "Corresponde a acumulación de km de un hueco de datos, no a un día real", int(excesivo.sum()))
    d = d.sort_values(["placa", "fecha"]).reset_index(drop=True)
    cob = d.groupby("placa")["fuente_km_dia"].value_counts(normalize=True).unstack(fill_value=0).round(3)
    registrar("km_dia_fuente", "proporción de días por fuente de km/día: " + str(cob.to_dict(orient="index")))
    ambos = d.dropna(subset=["km_dia_viajes", "km_dia_eventos"])
    ambos = ambos[ambos["estado_dia"] == "con viajes"]
    if len(ambos):
        r = ambos[["km_dia_viajes", "km_dia_eventos"]].corr().iloc[0, 1]
        registrar("km_dia_fuentes_concordancia", f"días con ambas fuentes: {len(ambos)} | correlación: {r:.3f} | "
                  f"diferencia mediana: {(ambos['km_dia_eventos'] - ambos['km_dia_viajes']).median():.1f} km")
    return d


def unir_ultimo_preventivo(diario, m, factor=None):
    """A cada camioneta-día le pega el último preventivo (y el siguiente real, como objetivo del pronóstico)."""
    prev = (m[~m["es_correctivo"]].drop_duplicates(["placa", "fecha"])
              .dropna(subset=["placa_eventos"])
              .rename(columns={"placa": "placa_hojas", "placa_eventos": "placa"})
              [["placa", "fecha", "km", "km_nominal"]].copy())
    prev["fecha"] = prev["fecha"].astype("datetime64[ns]")

    ult = prev.rename(columns={"fecha": "fecha_ult_prev", "km": "km_ult_prev", "km_nominal": "nominal_ult_prev"}).sort_values("fecha_ult_prev")
    sig = prev.rename(columns={"fecha": "fecha_prox_prev", "km": "km_prox_prev", "km_nominal": "nominal_prox_prev"}).sort_values("fecha_prox_prev")

    d = diario.copy()
    d["fecha"] = d["fecha"].astype("datetime64[ns]")
    d = d.sort_values("fecha")
    d = pd.merge_asof(d, ult, left_on="fecha", right_on="fecha_ult_prev", by="placa", direction="backward")
    d = pd.merge_asof(d, sig, left_on="fecha", right_on="fecha_prox_prev", by="placa",
                      direction="forward", allow_exact_matches=False)
    d = d.sort_values(["placa", "fecha"]).reset_index(drop=True)

    # El odómetro GPS queda por debajo del km del taller; se lleva a la escala del taller con el
    # factor por camioneta (ver factor_odometro). Se usa el odómetro de los viajes si existe ese día.
    odo_gps = d["odo_ini_viajes"].combine_first(d["odo_interp"]) if "odo_ini_viajes" in d else d["odo_interp"]
    d["factor_odo"] = d["placa"].map(factor) if factor is not None else 1.0
    d["km_hojas_estimado"] = odo_gps * d["factor_odo"]

    d["dias_desde_prev"] = (d["fecha"] - d["fecha_ult_prev"]).dt.days
    d["km_desde_prev"] = d["km_hojas_estimado"] - d["km_ult_prev"]
    d["km_nominal_proximo"] = _nominal_siguiente(d["nominal_ult_prev"], d["km_ult_prev"])
    d["km_al_proximo_nominal"] = d["km_nominal_proximo"] - d["km_hojas_estimado"]
    # Objetivo real del pronóstico (OA2): cuánto falta para el siguiente preventivo realmente ejecutado
    d["dias_al_proximo_real"] = (d["fecha_prox_prev"] - d["fecha"]).dt.days
    d["km_al_proximo_real"] = d["km_prox_prev"] - d["km_hojas_estimado"]
    d["objetivo_censurado"] = d["fecha_prox_prev"].isna()

    util = d["km_desde_prev"].notna()
    registrar("km_desde_prev_calidad", f"negativos (error residual del factor): {int((d['km_desde_prev'] < 0).sum())} | "
                                       f"> 12.000 km (ciclos largos o servicio omitido): {int((d['km_desde_prev'] > 12000).sum())}")
    registrar("diario_prev", f"{len(d)} camioneta-días | con km desde el último preventivo: {int(util.sum())} | "
                             f"con objetivo real (siguiente preventivo conocido): {int((~d['objetivo_censurado'] & util).sum())} | "
                             f"censurados: {int(d['objetivo_censurado'].sum())}")
    return d


def construir_ciclos(m, diario, ev):
    """Unidad de análisis de la Guía 2: camioneta x ciclo entre preventivos consecutivos."""
    vis = (m[~m["es_correctivo"]].drop_duplicates(["placa", "fecha"]).sort_values(["placa", "fecha"])
             [["placa", "placa_eventos", "fecha", "km", "km_nominal"]])
    corte = m["fecha"].max()              # el ciclo abierto se observa hasta el último registro de mantenimiento
    cols_alerta = [c for c in diario.columns if c.startswith("n_dlp_alerta")]
    cols_sin_cob = [c for c in ("n_dlp_no_gsm_coverage", "n_dlp_no_gprs_coverage") if c in diario.columns]
    filas = []
    for placa, g in vis.groupby("placa"):
        g = g.reset_index(drop=True)
        pe = g.loc[0, "placa_eventos"]
        for i in range(len(g)):
            a = g.loc[i]
            abierto = i + 1 >= len(g)
            b = None if abierto else g.loc[i + 1]
            fin = corte if abierto else b["fecha"]
            dias = (fin - a["fecha"]).days
            if dias <= 0:
                continue
            fila = {"placa_hojas": placa, "placa": pe, "fecha_ini": a["fecha"], "fecha_fin": fin, "abierto": abierto,
                    "dias": dias, "km_ini": a["km"], "km_fin": np.nan if abierto else b["km"],
                    "nominal_ini": a["km_nominal"], "nominal_fin": np.nan if abierto else b["km_nominal"]}
            fila["km_ciclo"] = fila["km_fin"] - fila["km_ini"]
            fila["km_dia_ciclo"] = fila["km_ciclo"] / dias if not abierto else np.nan

            # Correctivos dentro del ciclo (los del mismo día del preventivo final son parte de esa visita)
            fin_excl = (m["fecha"] <= fin) if abierto else (m["fecha"] < fin)
            cm = m[(m["placa"] == placa) & m["es_correctivo"] & (m["fecha"] > a["fecha"]) & fin_excl]
            npg = cm[cm["no_programado"]]
            fila["n_correctivos_intervalo"] = len(cm)
            fila["n_no_programados"] = len(npg)
            fila["falla_no_programada"] = len(npg) > 0
            fila["n_correctivos_en_visita_fin"] = 0 if abierto else int(((m["placa"] == placa) & m["es_correctivo"] & (m["fecha"] == fin)).sum())
            for s_ in SISTEMAS_CORRECTIVO:
                fila[f"nnp_{s_}"] = int(npg[f"corr_{s_}"].sum())

            # Telemetría del ciclo
            if isinstance(pe, str):
                dd = diario[(diario["placa"] == pe) & (diario["fecha"] >= a["fecha"]) & (diario["fecha"] < fin)]
                con_ev = int(dd["dia_con_eventos"].sum())
                fila["cobertura_tel"] = con_ev / dias
                fila["km_dia_tel"] = dd["km_dia"].mean()
                fila["volt_min"] = dd["voltaje_min"].min()
                fila["volt_medio"] = dd["voltaje_medio"].mean()
                fila["n_alertas_volt"] = dd[cols_alerta].sum().sum() if cols_alerta else np.nan
                fila["n_desconexion_bateria"] = dd["n_dlp_desconexion_bateria"].sum() if "n_dlp_desconexion_bateria" in dd else np.nan
                fila["n_sin_cobertura"] = dd[cols_sin_cob].sum().sum() if cols_sin_cob else np.nan
                fila["n_ralenti_excesivo"] = dd["n_ec_ralenti_excesivo"].sum() if "n_ec_ralenti_excesivo" in dd else np.nan
                fila["ralenti_h_dia"] = dd["duracion_ralenti_s"].sum() / 3600 / max(con_ev, 1)
                fila["pct_sin_gps"] = dd["n_sin_gps"].sum() / dd["n_eventos"].sum() if dd["n_eventos"].sum() else np.nan
                msk = (ev["placa"] == pe) & (ev["inicio"] >= a["fecha"]) & (ev["inicio"] < fin)
                fila["n_conductores"] = ev.loc[msk, "conductor"].nunique()
            filas.append(fila)
    ciclos = pd.DataFrame(filas)
    ciclos["ciclo_con_telemetria"] = ciclos["cobertura_tel"].fillna(0) >= COBERTURA_MIN_CICLO
    cerr = ciclos[~ciclos["abierto"]]
    registrar("ciclos", f"{len(ciclos)} ciclos ({len(cerr)} cerrados, {int(ciclos['abierto'].sum())} abiertos) | "
                        f"con telemetría >= {COBERTURA_MIN_CICLO:.0%}: {int(ciclos['ciclo_con_telemetria'].sum())} | "
                        f"con falla no programada: {int(ciclos['falla_no_programada'].sum())}")
    registrar("ciclos_km", f"km por ciclo (mediana): {cerr['km_ciclo'].median():.0f} | días (mediana): {cerr['dias'].median():.0f} | "
                           f"km/día (mediana): {cerr['km_dia_ciclo'].median():.1f}")
    return ciclos


# ---------- Técnicas de entendimiento que faltaban ----------
def reducir_dimensiones(diario):
    """Técnica 3: PCA sobre la telemetría diaria y lista de variables casi duplicadas (|r| > 0.9)."""
    cols = [c for c in diario.columns if c.startswith("n_") or c in ("odo_lecturas", "duracion_ralenti_s", "voltaje_min", "voltaje_medio", "km_dia")]
    X = diario[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.loc[:, X.notna().any()]
    X = X.loc[:, X.std(skipna=True) > 0]
    pct_nan = X.isna().mean().mean()
    Z = X.fillna(X.median())
    Z = (Z - Z.mean()) / Z.std(ddof=0)
    _, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    acum = np.cumsum(var)
    k90 = int(np.argmax(acum >= 0.90)) + 1
    cargas = pd.DataFrame(Vt[:3].T, index=X.columns, columns=["PC1", "PC2", "PC3"]).round(3)
    cargas.loc["varianza_explicada"] = var[:3].round(3)
    registrar("pca", f"{X.shape[1]} variables diarias | faltantes rellenados con la mediana: {pct_nan:.1%} | "
                     f"componentes para 90 % de la varianza: {k90} | PC1-3: {[round(float(v), 3) for v in var[:3]]}")
    r = X.corr().abs()
    pares = [(a, b, round(float(r.loc[a, b]), 3)) for i, a in enumerate(r.columns) for b in r.columns[i + 1:] if r.loc[a, b] > CORR_ALTA]
    pares = pd.DataFrame(pares, columns=["variable_a", "variable_b", "abs_r"]).sort_values("abs_r", ascending=False)
    registrar("correlaciones_altas", f"{len(pares)} pares con |r| > {CORR_ALTA}: " + "; ".join(f"{a}~{b}" for a, b, _ in pares.values[:6]))
    return cargas, pares


def importancia_variables(diario, ciclos):
    """Técnica 8 (exploratoria): ¿qué variables acompañan a la señal que se quiere vigilar?
    - OA3 (batería): correlación de Spearman, a nivel camioneta-día, entre el voltaje (y las alertas)
      y el uso del día (km, viajes cortos, horas de conducción, ralentí).
    - OA4 (riesgo): con 19 correctivos y 3 fallas no programadas no hay muestra para estimar
      importancias; solo se describe la relación entre uso del ciclo y número de correctivos.
    No se entrena ningún modelo: la Guía 4 pide entender los datos, no modelarlos, y con
    5 camionetas un modelo sobreajustaría (riesgo 7.2.1 de la Guía 3)."""
    d = diario.copy()
    cols_alerta = [c for c in d.columns if c.startswith("n_dlp_alerta")]
    d["n_alertas_voltaje"] = d[cols_alerta].sum(axis=1) if cols_alerta else np.nan
    d["ralenti_h"] = d.get("duracion_ralenti_s", np.nan) / 3600
    candidatas = ["km_dia", "n_viajes", "n_viajes_cortos", "horas_conduccion", "ralenti_h", "n_ec_ralenti_excesivo"]
    candidatas = [c for c in candidatas if c in d.columns]
    filas = []
    for objetivo in ("voltaje_min", "voltaje_medio", "n_alertas_voltaje"):
        base = d[d["dia_con_eventos"]].dropna(subset=[objetivo])
        for c in candidatas:
            par = base[[objetivo, c]].dropna()
            if len(par) > 30 and par[c].nunique() > 1:
                filas.append({"objetivo": objetivo, "variable": c, "spearman": round(par.corr(method="spearman").iloc[0, 1], 3), "n_dias": len(par)})
    out = pd.DataFrame(filas)
    cerr = ciclos[~ciclos["abierto"]].dropna(subset=["km_dia_ciclo"])
    if len(cerr) > 10:
        for c in ("km_dia_ciclo", "dias", "ralenti_h_dia"):
            if c in cerr and cerr[c].nunique() > 1:
                rho = cerr[[c, "n_correctivos_intervalo"]].corr(method="spearman").iloc[0, 1]
                out = pd.concat([out, pd.DataFrame([{"objetivo": "n_correctivos_ciclo", "variable": c, "spearman": round(rho, 3), "n_dias": len(cerr)}])])
    registrar("importancia", "Spearman (exploratorio, sin conclusiones causales): " +
              "; ".join(f"{r.objetivo}~{r.variable}={r.spearman:+.2f}" for r in out.itertuples() if abs(r.spearman) >= 0.2))
    return out.sort_values("spearman", key=lambda s_: s_.abs(), ascending=False).reset_index(drop=True)


def segmentar_camionetas(diario, ev, ciclos, m):
    """Técnica 7: perfil por camioneta y segmento de uso (alto / medio / bajo)."""
    rows = []
    cerr = ciclos[~ciclos["abierto"]]
    for pe, g in diario.groupby("placa"):
        e = ev[ev["placa"] == pe]
        c = cerr[cerr["placa"] == pe]
        corr = m[m["placa_eventos"].eq(pe) & m["es_correctivo"]]
        gv = g[g.get("fuente_km_dia", pd.Series("eventos", index=g.index)).eq("viajes")] if "fuente_km_dia" in g else g
        cols_alerta = [c for c in g.columns if c.startswith("n_dlp_alerta")]
        rows.append({"placa": pe, "dias_con_eventos": int(g["dia_con_eventos"].sum()),
                     "km_dia_mediana": g["km_dia"].median(), "km_dia_mediana_viajes": gv["km_dia"].median() if len(gv) else np.nan,
                     "cobertura_km_dia": g["km_dia"].notna().mean(),
                     "viajes_cortos_pct": g["n_viajes_cortos"].sum() / g["n_viajes"].sum() if "n_viajes" in g and g["n_viajes"].sum() else np.nan,
                     "alertas_voltaje": int(g[cols_alerta].sum().sum()) if cols_alerta else np.nan,
                     "desconexiones_bateria": int(g["n_dlp_desconexion_bateria"].sum()) if "n_dlp_desconexion_bateria" in g else np.nan,
                     "n_conductores": e["conductor"].nunique(), "pct_sin_gps": e["sin_gps_inicio"].mean(),
                     "ralenti_h_dia": g["duracion_ralenti_s"].sum() / 3600 / max(int(g["dia_con_eventos"].sum()), 1),
                     "volt_mediano_mv": g["voltaje_medio"].median(),
                     "dias_entre_prev_mediana": c["dias"].median(), "km_entre_prev_mediana": c["km_ciclo"].median(),
                     "n_correctivos": len(corr), "n_no_programados": int(corr["no_programado"].sum())})
    seg = pd.DataFrame(rows)
    seg["segmento_uso"] = np.select([seg["km_dia_mediana"] >= USO_ALTO_KM_DIA, seg["km_dia_mediana"] < USO_BAJO_KM_DIA], ["alto", "bajo"], "medio")
    registrar("segmentos", str(seg.set_index("placa")["segmento_uso"].to_dict()) +
              f" (alto >= {USO_ALTO_KM_DIA} km/día, bajo < {USO_BAJO_KM_DIA})")
    return seg.round(3)


def main():
    ev = limpiar_eventos(cargar_eventos())
    ev_dia = agregar_eventos_dia(ev)
    vj = cargar_viajes()
    vj_dia = None
    if vj is not None:
        vj = limpiar_viajes(vj)
        vj_dia = agregar_viajes_dia(vj)
    else:
        registrar("viajes", "no se encontró el Daily Movement Report anonimizado")
    diario = unir_fuentes_diarias(ev_dia, vj_dia)
    m, p, v = limpiar_hojas(cargar_hojas())
    cruce(ev, v, m)

    factor, cruces_odo = factor_odometro(vj_dia, ev_dia, m)
    diario_prev = unir_ultimo_preventivo(diario, m, factor)
    regla("tabla diaria", "Variables del día", "d", "Riesgo de fuga de información: usar datos posteriores a la fecha de la alerta",
          "construcción solo con información hasta el día (último preventivo hacia atrás); el próximo preventivo se guarda como objetivo aparte",
          "Una variable que se conoce después de la decisión da buenos resultados en la prueba pero no sirve en producción", len(diario_prev))
    ciclos = construir_ciclos(m, diario, ev)
    cargas, pares = reducir_dimensiones(diario)
    importancia = importancia_variables(diario, ciclos)
    segmentos = segmentar_camionetas(diario, ev, ciclos, m)
    RESULTADOS.update(ciclos=ciclos, diario_prev=diario_prev, pca_cargas=cargas, correlaciones_altas=pares,
                      importancia=importancia, segmentos=segmentos, viajes=vj, cruces_odometro=cruces_odo,
                      factor_odometro=factor)

    ev.to_csv(OUT_DIR / "eventos_limpios.csv", index=False)
    diario.to_csv(OUT_DIR / "tabla_diaria.csv", index=False)
    diario_prev.to_csv(OUT_DIR / "tabla_diaria_con_preventivo.csv", index=False)
    ciclos.to_csv(OUT_DIR / "ciclos_camioneta.csv", index=False)
    m.to_csv(OUT_DIR / "mantenimientos_limpios.csv", index=False)
    p.to_csv(OUT_DIR / "plan_limpio.csv", index=False)
    v.to_csv(OUT_DIR / "vehiculos.csv", index=False)
    if vj is not None:
        vj.to_csv(OUT_DIR / "viajes_limpios.csv", index=False)
    # Resultados agregados (pequeños, sin datos fila a fila): sí se versionan en docs/
    cargas.rename_axis("variable").to_csv(DOCS_DIR / "pca_telemetria.csv")
    pares.to_csv(DOCS_DIR / "correlaciones_altas.csv", index=False)
    importancia.to_csv(DOCS_DIR / "importancia_variables.csv", index=False)
    segmentos.to_csv(DOCS_DIR / "segmentos_camioneta.csv", index=False)
    cruces_odo.groupby("placa").agg(n=("factor", "size"), factor=("factor", "median"), sesgo_pct=("sesgo_pct", "median")).round(4).to_csv(DOCS_DIR / "factor_odometro.csv")
    pd.DataFrame(LOG).to_csv(DOCS_DIR / "log_preprocesamiento.csv", index=False)
    pd.DataFrame(REGLAS).to_csv(DOCS_DIR / "reglas_limpieza.csv", index=False)
    return ev, m, p, v, diario


if __name__ == "__main__":
    main()
