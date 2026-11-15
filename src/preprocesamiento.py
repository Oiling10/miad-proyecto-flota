"""
Pre-procesamiento de datos (Guía 4, Parte 3).
Entrada: data/anonimizados/ (solo lectura).
Salida: data/procesados/ (ignorada por Git) y docs/log_preprocesamiento.csv

Eventos (Detailed Event Report): decisiones validadas con el EDA.
Hojas de vida: pendiente de revisión con su propio EDA.
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
LOG = []

# Columnas del Detailed Event Report que se eliminan (AssetName se elimina al cargar)
COLS_ELIMINAR = [
    # 100 % vacías
    "Asset2ReportGroup", "FleetNumber", "RoadSpeedLimit", "EventLocation",
    "F_StartRegion", "F_EndRegion",
    # constantes o sin información
    "TripID", "FuelUsed", "DistanceCalculatedFromOdo", "EndOdo",
    "TotalOccurs", "TimeZone", "AssetSiteName", "ReportGroup",
    # redundante con el nombre del conductor
    "AssetExtra2",
    # calle y barrio: 63-84 % [UNKNOWN]
    "F_StartStreet", "F_StartSuburb", "F_EndStreet", "F_EndSuburb",
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
    for sistema, regla in REGLAS_SISTEMA.items():
        if re.search(regla, t):
            return sistema
    return "sin_clasificar"


# ---------- EVENTOS ----------
def cargar_eventos():
    ruta = IN_DIR / "Detailed Event Report_anonimizado.csv"
    ev = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    # Privacidad: AssetName trae la placa real; se descarta al cargar
    return ev.drop(columns=["AssetName"])


def limpiar_eventos(ev):
    ev = ev.drop(columns=COLS_ELIMINAR).copy()
    n0 = len(ev)
    registrar("columnas", f"eliminadas {len(COLS_ELIMINAR) + 1} (incluye AssetName) -> {ev.shape[1]} restantes")

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

    # Odómetro: 0.00 = no medido en ese tipo de evento (no se imputa por evento)
    ev["odo"] = _a_numero_con_coma_miles(ev["StartOdo"]).replace(0, np.nan)
    o = ev.loc[ev["odo"].notna()].sort_values(["placa", "inicio"])
    retro = int((o.groupby("placa")["odo"].diff() < 0).sum())
    registrar("odometro", f"lecturas reales: {len(o)} ({len(o) / len(ev):.1%}) | retrocesos: {retro}")

    # Voltaje (mV): solo existe en eventos de voltaje (nulo por diseño, no se imputa)
    ev["voltaje_mv"] = _a_numero_con_coma_miles(ev["EventValue"])
    ev["voltaje_cero"] = ev["voltaje_mv"].eq(0)
    registrar("voltaje", f"lecturas: {int(ev['voltaje_mv'].notna().sum())} | en 0 mV: {int(ev['voltaje_cero'].sum())}")

    # Coordenadas
    for suf, col in (("ini", "StartLatLong"), ("fin", "EndLatLong")):
        xy = ev[col].astype("string").str.split(",", expand=True)
        ev[f"lat_{suf}"] = pd.to_numeric(xy[0], errors="coerce")
        ev[f"lon_{suf}"] = pd.to_numeric(xy[1], errors="coerce")
        fuera = (~ev[f"lat_{suf}"].between(*LAT_RANGO) | ~ev[f"lon_{suf}"].between(*LON_RANGO)) & ev[f"lat_{suf}"].notna()
        ev.loc[fuera, [f"lat_{suf}", f"lon_{suf}"]] = np.nan
        registrar(f"coord_{suf}", f"fuera de rango: {int(fuera.sum())}")
    ev["sin_gps_inicio"] = ev["lat_ini"].isna()
    ev["sin_gps_fin"] = ev["lat_fin"].isna()
    # Inicio y fin casi coinciden (> 99 % a menos de 50 m): se completa uno con el otro, con marca
    ev["fin_imputado"] = ev["lat_fin"].isna() & ev["lat_ini"].notna()
    ev["ini_imputado"] = ev["lat_ini"].isna() & ev["lat_fin"].notna()
    for c in ("lat", "lon"):
        ev[f"{c}_fin"] = ev[f"{c}_fin"].fillna(ev[f"{c}_ini"])
        ev[f"{c}_ini"] = ev[f"{c}_ini"].fillna(ev[f"{c}_fin"])
    registrar("coord_imputadas", f"fin: {int(ev['fin_imputado'].sum())} | inicio: {int(ev['ini_imputado'].sum())} | sin ninguna: {int(ev['lat_ini'].isna().sum())}")

    # [UNKNOWN] = evento sin posición GPS -> faltante real
    ciudades = ["F_StartCity", "F_StartTown", "F_EndCity", "F_EndTown"]
    ev[ciudades] = ev[ciudades].replace("[UNKNOWN]", np.nan)
    ev = ev.rename(columns={"F_StartCity": "ciudad_ini", "F_StartTown": "municipio_ini",
                            "F_EndCity": "ciudad_fin", "F_EndTown": "municipio_fin",
                            "StartLocation": "geocerca_ini", "EndLocation": "geocerca_fin"})

    # Conductor sin identificar
    ev["conductor_sin_identificar"] = ev["conductor"].eq("Conductor no identificado")

    # Se descartan las columnas de texto ya convertidas
    ev = ev.drop(columns=["EventStartDate", "EventStartTime", "EventEndDate", "EventEndTime",
                          "TotalDuration", "StartOdo", "EventValue", "MeasurementUnits",
                          "StartLatLong", "EndLatLong"])

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
              .agg(voltaje_min="min", voltaje_medio="mean"))
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


# ---------- HOJAS DE VIDA ----------
# Las placas de las hojas de vida (VH-01 a VH-05) y las de eventos (VH-06 a VH-10) son la
# misma flota con códigos distintos. Equivalencia inferida por continuidad del odómetro
# (mantenimiento vs. eventos en la misma fecha). PENDIENTE DE CONFIRMAR con Gabriela.
EQUIVALENCIA_PLACAS = {"VH-01": "VH-09", "VH-02": "VH-10", "VH-03": "VH-07",
                       "VH-04": "VH-06", "VH-05": "VH-08"}

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
    v["fecha_placa"] = pd.to_datetime(v["fecha_placa"], errors="coerce")
    v["antiguedad_dias"] = (fecha_corte - v["fecha_placa"]).dt.days
    registrar("vehiculos", f"{len(v)} filas x {v.shape[1]} columnas (antigüedad al {fecha_corte.date()})")
    return v


def limpiar_mantenimientos(mant):
    m = mant.rename(columns={
        "Placa": "placa", "Fecha": "fecha", "Descripción": "descripcion", "Tipo": "tipo",
        "Kilometraje acumulado": "km", "Orden de servicio": "orden",
        "Costo (COP)": "costo", "Prestador del servicio": "prestador"}).copy()
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

    # Correctivos: sistemas afectados (varias banderas por descripción)
    corr = m["es_correctivo"]
    for sistema, regla in SISTEMAS_CORRECTIVO.items():
        m[f"corr_{sistema}"] = corr & m["descripcion"].str.lower().str.contains(regla, regex=True)
    cols_sis = [f"corr_{s}" for s in SISTEMAS_CORRECTIVO]
    sin_sis = int((corr & ~m[cols_sis].any(axis=1)).sum())
    registrar("correctivos", f"{int(corr.sum())} | sin sistema clasificado: {sin_sis} | "
                             + str({s: int(m[f'corr_{s}'].sum()) for s in SISTEMAS_CORRECTIVO}))
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
    return p


def limpiar_hojas(hv):
    fecha_corte = pd.to_datetime(hv["Mantenimientos"]["Fecha"], errors="coerce").max()
    v = limpiar_vehiculos(hv["Vehículos"], fecha_corte)
    m = limpiar_mantenimientos(hv["Mantenimientos"])
    p = limpiar_plan(hv["Plan de mantenimiento"], m)

    # Equivalencia con las placas de eventos (se conserva la placa original de hojas de vida)
    for df in (v, m, p):
        df["placa_eventos"] = df["placa"].map(EQUIVALENCIA_PLACAS)
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


def unir_ultimo_preventivo(diario, m):
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

    # Las dos fuentes de km (hojas vs. eventos) difieren 400-1.500 km: se estima un desfase por camioneta
    # con los días en que hubo mantenimiento y odómetro de eventos, y se expresa el km de eventos en la escala de las hojas.
    coinc = d.merge(prev[["placa", "fecha", "km"]], on=["placa", "fecha"], how="inner").dropna(subset=["odo_interp"])
    offset = (coinc["odo_interp"] - coinc["km"]).groupby(coinc["placa"]).median()
    d["offset_odo"] = d["placa"].map(offset)
    d["km_hojas_estimado"] = d["odo_interp"] - d["offset_odo"]
    registrar("desfase_km", "mediana (km eventos - km hojas) por camioneta: " + str({k: int(v) for k, v in offset.items()}))

    d["dias_desde_prev"] = (d["fecha"] - d["fecha_ult_prev"]).dt.days
    d["km_desde_prev"] = d["km_hojas_estimado"] - d["km_ult_prev"]
    d["km_nominal_proximo"] = _nominal_siguiente(d["nominal_ult_prev"], d["km_ult_prev"])
    d["km_al_proximo_nominal"] = d["km_nominal_proximo"] - d["km_hojas_estimado"]
    # Objetivo real del pronóstico (OA2): cuánto falta para el siguiente preventivo realmente ejecutado
    d["dias_al_proximo_real"] = (d["fecha_prox_prev"] - d["fecha"]).dt.days
    d["km_al_proximo_real"] = d["km_prox_prev"] - d["km_hojas_estimado"]
    d["objetivo_censurado"] = d["fecha_prox_prev"].isna()

    util = d["km_desde_prev"].notna()
    registrar("km_desde_prev_calidad", f"negativos (error del desfase, <= 400 km): {int((d['km_desde_prev'] < 0).sum())} | "
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


def importancia_variables(ciclos):
    """Técnica 8 (exploratoria): ¿qué variables de telemetría acompañan al uso (km/día) de cada ciclo?"""
    base = ciclos[ciclos["ciclo_con_telemetria"] & ~ciclos["abierto"]].dropna(subset=["km_dia_ciclo"]).copy()
    feats = ["km_dia_tel", "volt_min", "volt_medio", "n_alertas_volt", "n_desconexion_bateria", "n_sin_cobertura",
             "n_ralenti_excesivo", "ralenti_h_dia", "pct_sin_gps", "n_conductores", "n_correctivos_intervalo"]
    feats = [c for c in feats if c in base.columns and base[c].notna().sum() > 3 and base[c].nunique() > 1]
    sp = base[feats + ["km_dia_ciclo"]].corr(method="spearman")["km_dia_ciclo"].drop("km_dia_ciclo")
    out = pd.DataFrame({"spearman_con_km_dia_ciclo": sp.round(3), "n": [int(base[c].notna().sum()) for c in sp.index]})
    out["importancia_rf"] = np.nan
    msg = f"ciclos usados: {len(base)} (exploratorio, sin conclusiones causales)"
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GroupKFold, cross_val_predict
        fs = [c for c in feats if c != "km_dia_tel"]          # km_dia_tel es casi el mismo dato que el objetivo
        Xr, y = base[fs].fillna(base[fs].median()), base["km_dia_ciclo"]
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0)
        out.loc[fs, "importancia_rf"] = rf.fit(Xr, y).feature_importances_.round(3)
        if base["placa"].nunique() >= 3:
            pred = cross_val_predict(rf, Xr, y, cv=GroupKFold(n_splits=min(5, base["placa"].nunique())), groups=base["placa"])
            mae_rf, mae_med = float(np.abs(y - pred).mean()), float(np.abs(y - y.median()).mean())
            msg += f" | MAE validación por camioneta: RF {mae_rf:.1f} vs mediana {mae_med:.1f} km/día"
    except ImportError:
        msg += " | scikit-learn no disponible: solo correlaciones"
    registrar("importancia", msg)
    registrar("importancia_top", str(out["spearman_con_km_dia_ciclo"].abs().sort_values(ascending=False).head(4).round(2).to_dict()))
    return out.sort_values("spearman_con_km_dia_ciclo", key=lambda s: s.abs(), ascending=False)


def segmentar_camionetas(diario, ev, ciclos, m):
    """Técnica 7: perfil por camioneta y segmento de uso (alto / medio / bajo)."""
    rows = []
    cerr = ciclos[~ciclos["abierto"]]
    for pe, g in diario.groupby("placa"):
        e = ev[ev["placa"] == pe]
        c = cerr[cerr["placa"] == pe]
        corr = m[m["placa_eventos"].eq(pe) & m["es_correctivo"]]
        rows.append({"placa": pe, "dias_con_eventos": int(g["dia_con_eventos"].sum()),
                     "km_dia_mediana": g["km_dia"].median(), "cobertura_km_dia": g["km_dia"].notna().mean(),
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
    diario = agregar_eventos_dia(ev)
    m, p, v = limpiar_hojas(cargar_hojas())
    cruce(ev, v, m)

    # Etiqueta, ciclos y unión con el último preventivo
    diario_prev = unir_ultimo_preventivo(diario, m)
    ciclos = construir_ciclos(m, diario, ev)
    # Técnicas de entendimiento
    cargas, pares = reducir_dimensiones(diario)
    importancia = importancia_variables(ciclos)
    segmentos = segmentar_camionetas(diario, ev, ciclos, m)
    RESULTADOS.update(ciclos=ciclos, diario_prev=diario_prev, pca_cargas=cargas, correlaciones_altas=pares,
                      importancia=importancia, segmentos=segmentos)

    ev.to_csv(OUT_DIR / "eventos_limpios.csv", index=False)
    diario.to_csv(OUT_DIR / "eventos_por_placa_dia.csv", index=False)
    diario_prev.to_csv(OUT_DIR / "tabla_diaria_con_preventivo.csv", index=False)
    ciclos.to_csv(OUT_DIR / "ciclos_camioneta.csv", index=False)
    m.to_csv(OUT_DIR / "mantenimientos_limpios.csv", index=False)
    p.to_csv(OUT_DIR / "plan_limpio.csv", index=False)
    v.to_csv(OUT_DIR / "vehiculos.csv", index=False)
    # Resultados agregados de las técnicas (pequeños, sin datos crudos): sí se versionan en docs/
    cargas.rename_axis("variable").to_csv(DOCS_DIR / "pca_telemetria.csv")
    pares.to_csv(DOCS_DIR / "correlaciones_altas.csv", index=False)
    importancia.rename_axis("variable").to_csv(DOCS_DIR / "importancia_variables.csv")
    segmentos.to_csv(DOCS_DIR / "segmentos_camioneta.csv", index=False)
    pd.DataFrame(LOG).to_csv(DOCS_DIR / "log_preprocesamiento.csv", index=False)
    return ev, m, p, v, diario


if __name__ == "__main__":
    main()
