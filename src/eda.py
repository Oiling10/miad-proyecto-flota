"""
eda.py - Funciones de apoyo para el EDA del proyecto "Predicción de fallas en flotas con ML".


    1. Instalación y lectura ........ verificar_entorno, localizar_datos, rutas_datos, leer_consolidado,
                                      leer_eventos, colores, configurar_estilo, mostrar_tablas
    2. EDA del consolidado .......... resumen_general, preparar_consolidado, tabla_completitud, reporte_completitud,
                                      reporte_duplicados_consolidado, integridad_placas, cruce_mtto_plan,
                                      perfil_vehiculos, resumen_fechas_mtto, resumen_tipo_mtto, intervalos_mtto,
                                      resumen_costos, grafico_costos, resumen_orden_prestador, resumen_visitas,
                                      grafico_linea_tiempo_mtto
    3. EDA del Detailed Event Report  convertir_tipos_eventos, valores_centinela, voltaje_en_cero, reporte_duplicados_eventos,
                                      cortes_red_gemelos, revisar_confidencialidad, eliminar_identificadores,
                                      equivalencia_placas_mascarada, cobertura_temporal, hueco_mas_largo,
                                      grafico_eventos_mensuales, catalogo_eventos, resumen_eventos_bateria,
                                      resumen_duracion, grafico_duracion, resumen_ralenti, grafico_ralenti,
                                      resumen_voltaje, grafico_voltaje_diario, serie_voltaje_mensual,
                                      correctivos_bateria, grafico_bateria_camioneta, resumen_odometro,
                                      grafico_odometro, resumen_conductores, resumen_ubicacion, resumen_red,
                                      comparar_llaves

Convenciones:
    - "hojas" es el diccionario {"Vehículos": df, "Mantenimientos": df, "Plan de mantenimiento": df}.
    - "ev" es el DataFrame de eventos ya convertido con convertir_tipos_eventos().
    - Las funciones gráficas devuelven la figura (ya cerrada para que no se dibuje dos veces); en el notebook
      basta con dejar la figura como última línea de la celda.
"""


import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display


# =====================================================================================================
# 1. INSTALACIÓN Y LECTURA
# =====================================================================================================
def verificar_entorno():
    """Imprime y devuelve las versiones de Python y de las librerías principales.

    Sirve para confirmar que el notebook corre con el entorno virtual correcto (.venv).

    Devuelve
    -------
    dict : {"python": str, "pandas": str, "numpy": str, "matplotlib": str}
    """
    info = {"python": sys.version.split()[0], "pandas": pd.__version__,
            "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    print(" | ".join(f"{k} {v}" for k, v in info.items()))
    return info


def localizar_datos():
    """Encuentra la carpeta data/anonimizados sin importar si el notebook corre desde notebooks/ o desde la raíz.

    Devuelve
    -------
    pathlib.Path : ruta a data/anonimizados.

    Lanza
    -----
    FileNotFoundError : si no encuentra la carpeta en ninguna de las dos ubicaciones.
    """
    for p in (Path("..") / "data" / "anonimizados", Path("data") / "anonimizados"):
        if p.exists():
            return p
    raise FileNotFoundError("No se encontró data/anonimizados. Ejecuta el notebook desde la carpeta notebooks/ del repo.")


def rutas_datos(data=None):
    """Arma las rutas de los dos archivos de entrada y muestra su tamaño en MB.

    Parámetros
    ----------
    data : str | Path | None
        Carpeta de datos. Si es None se usa localizar_datos().

    Devuelve
    -------
    (Path, Path) : (ruta del consolidado .xlsx, ruta del Detailed Event Report .csv)
    """
    data = Path(data) if data else localizar_datos()
    f_cons = data / "Consolidado_Hojas_de_Vida_anonimizado.xlsx"
    f_ev = data / "Detailed Event Report_anonimizado.csv"
    for f in (f_cons, f_ev):
        print(f"{f.name:55s} {f.stat().st_size / 1e6:8.1f} MB")
    return f_cons, f_ev


def leer_consolidado(ruta):
    """Lee las tres hojas del consolidado de hojas de vida TAL COMO VIENEN (sin corregir tipos).

    Parámetros
    ----------
    ruta : str | Path
        Ruta del archivo Consolidado_Hojas_de_Vida_anonimizado.xlsx.

    Devuelve
    -------
    dict[str, DataFrame] : claves "Vehículos", "Mantenimientos" y "Plan de mantenimiento".
    """
    return pd.read_excel(ruta, sheet_name=None)


def leer_eventos(ruta):
    """Lee el Detailed Event Report (CSV de ~96 MB) TAL COMO VIENE, sin corregir tipos.

    Nota: StartOdo y EventValue llegan como texto porque traen coma de miles; se corrigen en
    convertir_tipos_eventos().

    Parámetros
    ----------
    ruta : str | Path
        Ruta del archivo Detailed Event Report_anonimizado.csv.

    Devuelve
    -------
    DataFrame : eventos crudos.
    """
    return pd.read_csv(ruta, low_memory=False)


def colores():
    """Paleta del EDA: gris para los datos, un solo acento (rojo) para lo que importa.

    Devuelve
    -------
    dict : {"gris", "acento", "tinta"} con colores hexadecimales.
    """
    return {"gris": "#9aa0a6", "acento": "#d6452d", "tinta": "#3c4043"}


def configurar_estilo():
    """Configura opciones de pandas y el estilo de las gráficas (limpio, rejilla suave, sin bordes superiores).

    Se llama una sola vez al inicio del notebook.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)
    c = colores()
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#bdc1c6", "axes.labelcolor": c["tinta"], "xtick.color": c["tinta"],
        "ytick.color": c["tinta"], "axes.grid": True, "grid.color": "#e8eaed", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def mostrar_tablas(tablas, titulos=True):
    """Muestra en el notebook todas las tablas y valores de un diccionario de resultados.

    Parámetros
    ----------
    tablas : dict
        Resultado típico de las funciones resumen_*: valores sueltos (se imprimen) y DataFrames/Series (se muestran).
    titulos : bool
        Si es True imprime el nombre de cada elemento antes de mostrarlo.
    """
    for nombre, obj in tablas.items():
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            if titulos:
                print(f"\n-- {nombre}")
            display(obj)
        else:
            print(f"{nombre}: {obj}")


# =====================================================================================================
# 2. EDA DEL CONSOLIDADO DE HOJAS DE VIDA
# =====================================================================================================
def resumen_general(d, ocultar=None):
    """Tamaño y tipos de dato de una tabla.

    Parámetros
    ----------
    d : DataFrame
    ocultar : list[str] | None
        Columnas cuyo ejemplo no debe mostrarse (identificadores, por confidencialidad).

    Devuelve
    -------
    dict : {"filas", "columnas", "memoria_mb", "tipos"}; "tipos" es un DataFrame con dtype y un valor de ejemplo.
    """
    tipos = pd.DataFrame({"dtype": d.dtypes.astype(str),
                          "ejemplo": d.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None)})
    if ocultar:
        tipos.loc[[c for c in ocultar if c in tipos.index], "ejemplo"] = "(oculto, ver confidencialidad)"
    return {"filas": d.shape[0], "columnas": d.shape[1],
            "memoria_mb": round(d.memory_usage(deep=True).sum() / 1e6, 2), "tipos": tipos}


def preparar_consolidado(hojas):
    """Corrige tipos y crea las columnas derivadas del consolidado (trabaja sobre copias).

    Crea:
        Mantenimientos: Fecha (datetime), es_prev, es_corr (booleanos según la columna Tipo).
        Plan: Fecha programada/ejecutada (datetime), ejecutado (bool), desv_dias y desv_km
              (ejecutado menos programado). Los nulos de estas dos últimas heredan los del plan.

    Parámetros
    ----------
    hojas : dict[str, DataFrame]
        Resultado de leer_consolidado().

    Devuelve
    -------
    dict[str, DataFrame] : mismas tres hojas, con tipos corregidos y columnas derivadas.
    """
    veh = hojas["Vehículos"].copy()
    mtto = hojas["Mantenimientos"].copy()
    plan = hojas["Plan de mantenimiento"].copy()
    mtto["Fecha"] = pd.to_datetime(mtto["Fecha"], errors="coerce")
    mtto["es_prev"] = mtto["Tipo"].str.upper().str.contains("PREVENT", na=False)
    mtto["es_corr"] = mtto["Tipo"].str.upper().str.contains("CORRECT", na=False)
    for c in ["Fecha programada", "Fecha ejecutada"]:
        plan[c] = pd.to_datetime(plan[c], errors="coerce")
    plan["ejecutado"] = plan["Fecha ejecutada"].notna()
    plan["desv_dias"] = (plan["Fecha ejecutada"] - plan["Fecha programada"]).dt.days
    plan["desv_km"] = plan["Km ejecutado"] - plan["Km programado"]
    return {"Vehículos": veh, "Mantenimientos": mtto, "Plan de mantenimiento": plan}


def tabla_completitud(d):
    """Nulos, porcentaje de nulos y número de valores únicos por columna.

    Devuelve
    -------
    DataFrame : índice = columna; columnas "nulos", "% nulos", "únicos".
    """
    return pd.DataFrame({"nulos": d.isna().sum(), "% nulos": (d.isna().mean() * 100).round(1),
                         "únicos": d.nunique(dropna=True)})


def reporte_completitud(d, excluir_constantes=False):
    """Resume la completitud de una tabla: columnas vacías, constantes y columnas con nulos.

    Parámetros
    ----------
    d : DataFrame
    excluir_constantes : bool
        Si es True, la tabla "con_nulos" omite las columnas vacías y las constantes (deja solo las informativas).

    Devuelve
    -------
    dict : {"vacias": list, "constantes": list, "con_nulos": DataFrame o texto "Sin nulos"}
    """
    c = tabla_completitud(d)
    vacias = c.index[c["% nulos"] == 100].tolist()
    constantes = c.index[(c["únicos"] == 1) & (c["% nulos"] < 100)].tolist()
    con_nulos = c[c["nulos"] > 0]
    if excluir_constantes:
        con_nulos = con_nulos[(con_nulos["% nulos"] < 100) & ~con_nulos.index.isin(constantes)]
    return {"vacias": vacias, "constantes": constantes,
            "con_nulos": con_nulos if len(con_nulos) else "Sin nulos"}


def reporte_duplicados_consolidado(hojas):
    """Busca filas completas duplicadas y llaves repetidas en el consolidado.

    Llaves revisadas: Placa (Vehículos); Orden de servicio y Placa+Fecha+Descripción (Mantenimientos);
    Placa+Año+Km programado (Plan). Una Orden de servicio repetida no es error: agrupa visitas con
    preventivo y correctivo.

    Parámetros
    ----------
    hojas : dict[str, DataFrame]  (preferiblemente el resultado de preparar_consolidado)

    Devuelve
    -------
    dict : {"filas_completas": dict hoja -> n duplicadas, "llaves": DataFrame con filas de llave repetida}
    """
    llaves = {
        "Vehículos · Placa": ("Vehículos", ["Placa"]),
        "Mantenimientos · Orden de servicio": ("Mantenimientos", ["Orden de servicio"]),
        "Mantenimientos · Placa+Fecha+Descripción": ("Mantenimientos", ["Placa", "Fecha", "Descripción"]),
        "Plan · Placa+Año+Km programado": ("Plan de mantenimiento", ["Placa", "Año del plan", "Km programado"]),
    }
    return {
        "filas_completas": {n: int(d.duplicated().sum()) for n, d in hojas.items()},
        "llaves": pd.DataFrame([{"llave": n, "filas con llave repetida": int(hojas[h].duplicated(c, keep=False).sum())}
                                for n, (h, c) in llaves.items()]),
    }


def integridad_placas(hojas):
    """Verifica que las tres hojas hablen de las mismas camionetas y que la placa tenga el formato VH-NN.

    Devuelve
    -------
    dict : {"presencia": DataFrame (placa x hoja + n.º de registros en Mantenimientos y Plan),
            "formato": DataFrame (por hoja: formato inválido y placas con espacios)}
    """
    placas = {n: set(d["Placa"].dropna().astype(str)) for n, d in hojas.items()}
    todas = sorted(set().union(*placas.values()))
    pres = pd.DataFrame({n: [p in s for p in todas] for n, s in placas.items()}, index=todas)
    pres["registros_mtto"] = hojas["Mantenimientos"]["Placa"].value_counts().reindex(todas).fillna(0).astype(int)
    pres["filas_plan"] = hojas["Plan de mantenimiento"]["Placa"].value_counts().reindex(todas).fillna(0).astype(int)
    filas = []
    for n, d in hojas.items():
        s = d["Placa"].astype("string")
        filas.append({"hoja": n, "formato_invalido": int((~s.str.fullmatch(r"VH-\d{2}").fillna(False)).sum()),
                      "con_espacios": int((s != s.str.strip()).sum())})
    return {"presencia": pres, "formato": pd.DataFrame(filas).set_index("hoja")}


def cruce_mtto_plan(mtto, plan):
    """Cruza los preventivos de Mantenimientos con los preventivos ejecutados del Plan (misma placa y fecha).

    Parámetros
    ----------
    mtto, plan : DataFrame (resultado de preparar_consolidado)

    Devuelve
    -------
    dict : {"conteo": Series (en ambas / solo en Mantenimientos / solo en Plan),
            "n_ambas": int, "n_km_distinto": int,
            "km_distinto": DataFrame (coinciden en fecha pero no en km),
            "sin_pareja": DataFrame (registros sin contraparte exacta)}
    """
    a = mtto[mtto["es_prev"]][["Placa", "Fecha", "Kilometraje acumulado", "Descripción"]]
    b = plan[plan["ejecutado"]][["Placa", "Fecha ejecutada", "Km ejecutado", "Texto original ejecutado"]]
    cr = a.merge(b, left_on=["Placa", "Fecha"], right_on=["Placa", "Fecha ejecutada"], how="outer", indicator=True)
    etiqueta = cr["_merge"].map({"both": "en ambas", "left_only": "solo en Mantenimientos", "right_only": "solo en Plan"})
    ambas = cr[cr["_merge"] == "both"].copy()
    ambas["dif_km"] = ambas["Kilometraje acumulado"] - ambas["Km ejecutado"]
    dist = ambas[ambas["dif_km"].abs() > 0]
    return {
        "conteo": etiqueta.astype(str).value_counts(),
        "n_ambas": len(ambas), "n_km_distinto": len(dist),
        "km_distinto": dist[["Placa", "Fecha", "Kilometraje acumulado", "Km ejecutado", "dif_km"]],
        "sin_pareja": cr[cr["_merge"] != "both"][["Placa", "Fecha", "Fecha ejecutada", "Kilometraje acumulado",
                                                 "Km ejecutado", "_merge"]],
    }


def perfil_vehiculos(veh):
    """Perfil de la flota de la muestra: qué columnas varían entre camionetas y cuáles son constantes.

    Devuelve
    -------
    dict : {"columnas_que_varian": list, "n_constantes_o_vacias": int,
            "perfil": DataFrame (placa, marca, línea, modelo, clase, combustible),
            "detalle": DataFrame (columnas que varían, una columna por camioneta)}
    """
    informativas = [c for c in veh.columns if veh[c].nunique(dropna=True) > 1]
    return {"columnas_que_varian": informativas, "n_constantes_o_vacias": veh.shape[1] - len(informativas),
            "perfil": veh[["Placa", "Marca", "Línea", "Modelo", "Clase", "Combustible"]],
            "detalle": veh[informativas].set_index("Placa").T}


def resumen_fechas_mtto(mtto, hoy=None):
    """Rango de fechas de Mantenimientos, fechas futuras y primer/último registro por camioneta.

    Parámetros
    ----------
    hoy : Timestamp | None  (si es None se usa la fecha actual)

    Devuelve
    -------
    dict : {"primera", "ultima", "fechas_futuras", "por_placa": DataFrame}
    """
    hoy = pd.Timestamp.today().normalize() if hoy is None else pd.Timestamp(hoy)
    return {"primera": mtto["Fecha"].min().date(), "ultima": mtto["Fecha"].max().date(),
            "fechas_futuras": int((mtto["Fecha"] > hoy).sum()),
            "por_placa": mtto.groupby("Placa")["Fecha"].agg(primera="min", ultima="max", registros="count")}


def resumen_tipo_mtto(mtto):
    """Distribución preventivo/correctivo y consistencia de la columna Tipo con el texto de la Descripción.

    Devuelve
    -------
    dict : {"conteo_tipo": Series, "pct_correctivos": float, "por_placa": DataFrame,
            "consistencia": DataFrame (Tipo vs lo que dice la descripción)}
    """
    desc = mtto["Descripción"].astype("string").str.upper().str.strip()
    dice = np.select([desc.str.contains("PREVENT", na=False), desc.str.contains("CORRECT", na=False)],
                     ["dice PREVENTIVO", "dice CORRECTIVO"], default="no dice")
    return {"conteo_tipo": mtto["Tipo"].value_counts(dropna=False),
            "pct_correctivos": round(float(mtto["es_corr"].mean() * 100), 1),
            "por_placa": pd.crosstab(mtto["Placa"], mtto["Tipo"], margins=True),
            "consistencia": pd.crosstab(mtto["Tipo"], dice, rownames=["Tipo (columna)"],
                                        colnames=["Descripción (texto)"])}


def intervalos_mtto(mtto):
    """Orden cronológico, retrocesos de km e intervalos entre eventos y entre preventivos consecutivos.

    Devuelve
    -------
    dict : {"archivo_en_orden": bool, "retrocesos_km": int, "km_repetido": int,
            "sin_avance": DataFrame (registros con km igual o menor al anterior),
            "km_por_placa": DataFrame, "entre_preventivos": DataFrame (describe de d_km y d_dias),
            "m": DataFrame ordenado con d_km/d_dias, "mp": solo preventivos con d_km/d_dias}
    """
    m = mtto.sort_values(["Placa", "Fecha", "Kilometraje acumulado"]).copy()
    m["d_km"] = m.groupby("Placa")["Kilometraje acumulado"].diff()
    m["d_dias"] = m.groupby("Placa")["Fecha"].diff().dt.days
    mp = m[m["es_prev"]].copy()
    mp["d_km"] = mp.groupby("Placa")["Kilometraje acumulado"].diff()
    mp["d_dias"] = mp.groupby("Placa")["Fecha"].diff().dt.days
    return {
        "archivo_en_orden": bool(mtto.groupby("Placa")["Fecha"].apply(lambda s: s.is_monotonic_increasing).all()),
        "retrocesos_km": int((m["d_km"] < 0).sum()), "km_repetido": int((m["d_km"] == 0).sum()),
        "sin_avance": m.loc[m["d_km"] <= 0, ["Placa", "Fecha", "Descripción", "Kilometraje acumulado", "d_km", "d_dias"]],
        "km_por_placa": m.groupby("Placa")["Kilometraje acumulado"].agg(["min", "max", "count"]),
        "entre_preventivos": mp[["d_km", "d_dias"]].describe().round(0),
        "m": m, "mp": mp,
    }


def resumen_costos(mtto):
    """Calidad y distribución de los costos por tipo de mantenimiento, con atípicos por regla IQR.

    Devuelve
    -------
    dict : {"nulos", "ceros", "negativos", "por_tipo": describe, "total_por_tipo": DataFrame,
            "pct_correctivos_del_gasto": float, "n_atipicos": int, "atipicos": DataFrame}
    """
    costo = mtto["Costo (COP)"]

    def atipico(g):
        q1, q3 = g.quantile([.25, .75])
        return g > q3 + 1.5 * (q3 - q1)

    flag = mtto.groupby("Tipo")["Costo (COP)"].transform(atipico)
    total = mtto.groupby("Tipo")["Costo (COP)"].sum()
    corr = total[[t for t in total.index if "CORREC" in t.upper()]].sum()
    return {"nulos": int(costo.isna().sum()), "ceros": int((costo == 0).sum()), "negativos": int((costo < 0).sum()),
            "por_tipo": mtto.groupby("Tipo")["Costo (COP)"].describe().round(0),
            "total_por_tipo": total.to_frame("total COP"),
            "pct_correctivos_del_gasto": round(float(corr / total.sum() * 100), 1),
            "n_atipicos": int(flag.sum()),
            "atipicos": mtto.loc[flag, ["Placa", "Fecha", "Descripción", "Tipo", "Costo (COP)"]]}


def grafico_costos(mtto):
    """Gráfico de puntos del costo de cada registro por tipo (escala logarítmica); correctivos en rojo.

    Devuelve
    -------
    matplotlib.figure.Figure
    """
    c = colores()
    fig, ax = plt.subplots(figsize=(8, 2.6))
    tipos = sorted(mtto["Tipo"].unique())
    for i, t in enumerate(tipos):
        g = mtto[mtto["Tipo"] == t]
        jit = np.random.default_rng(0).uniform(-0.12, 0.12, len(g))
        ax.scatter(g["Costo (COP)"] / 1e6, np.full(len(g), i) + jit, s=22,
                   color=c["acento"] if "CORREC" in t.upper() else c["gris"], edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(tipos)))
    ax.set_yticklabels(tipos)
    ax.set_xscale("log")
    ax.set_xlabel("Costo por registro (millones de COP, escala log)")
    ax.set_title("Costo de cada registro de mantenimiento por tipo")
    ax.grid(axis="y", visible=False)
    plt.close(fig)
    return fig


def resumen_orden_prestador(mtto):
    """Calidad de la Orden de servicio y reparto de registros entre prestadores.

    Devuelve
    -------
    dict : {"os_nulas", "os_unicas", "total_registros", "formato_OS_NNN": bool,
            "prestador_por_tipo": DataFrame}
    """
    os_ = mtto["Orden de servicio"]
    return {"os_nulas": int(os_.isna().sum()), "os_unicas": int(os_.nunique()), "total_registros": len(mtto),
            "formato_OS_NNN": bool(os_.astype("string").str.fullmatch(r"OS-\d{3}").all()),
            "prestador_por_tipo": pd.crosstab(mtto["Prestador del servicio"], mtto["Tipo"], margins=True)}


def resumen_visitas(mtto, umbral_dias=3):
    """Visitas (placa+fecha) con más de un registro y regla de falla no programada de la Guía 2.

    Un correctivo es "no programado" si está a más de `umbral_dias` días de un preventivo de la misma camioneta.
    OJO: el km de los correctivos suele heredarse del preventivo cercano y la Fecha puede ser la del servicio,
    no la de la falla.

    Parámetros
    ----------
    umbral_dias : int  (por defecto 3, según la Guía 2)

    Devuelve
    -------
    dict : {"n_visitas", "visitas_con_mas_de_un_registro", "tipos_de_visita": DataFrame,
            "correctivos": DataFrame (con min_dias_a_preventivo y no_programado),
            "n_no_programados": int, "sensibilidad": DataFrame (n.º de no programados según el umbral)}
    """
    visitas = mtto.groupby(["Placa", "Fecha"]).agg(
        n=("Tipo", "size"), tipos=("Tipo", lambda s: " + ".join(sorted(set(s)))))
    prev = mtto[mtto["es_prev"]][["Placa", "Fecha"]].rename(columns={"Fecha": "fecha_prev"})
    corr = mtto[mtto["es_corr"]][["Placa", "Fecha", "Descripción", "Kilometraje acumulado"]].reset_index(names="idx")
    x = corr.merge(prev, on="Placa")
    x["dias_a_prev"] = (x["Fecha"] - x["fecha_prev"]).abs().dt.days
    corr["min_dias_a_preventivo"] = corr["idx"].map(x.groupby("idx")["dias_a_prev"].min())
    corr["no_programado"] = corr["min_dias_a_preventivo"] > umbral_dias
    sens = pd.DataFrame({"umbral (días)": [0, 1, 3, 5, 7, 10, 30]})
    sens["no programados"] = [int((corr["min_dias_a_preventivo"] > u).sum()) for u in sens["umbral (días)"]]
    sens["de correctivos"] = len(corr)
    return {"n_visitas": len(visitas), "visitas_con_mas_de_un_registro": int((visitas["n"] > 1).sum()),
            "tipos_de_visita": visitas["tipos"].value_counts().to_frame("visitas"),
            "correctivos": corr.drop(columns="idx"), "n_no_programados": int(corr["no_programado"].sum()),
            "sensibilidad": sens}


def grafico_linea_tiempo_mtto(mtto):
    """Kilometraje acumulado por camioneta en el tiempo (una fila por camioneta).

    Gris: preventivos y línea de kilometraje. Rojo: correctivos (su km se hereda del preventivo cercano).

    Devuelve
    -------
    matplotlib.figure.Figure
    """
    c = colores()
    m = mtto.sort_values(["Placa", "Fecha", "Kilometraje acumulado"])
    placas = sorted(m["Placa"].unique())
    fig, axes = plt.subplots(len(placas), 1, sharex=True, figsize=(10, 1.9 * len(placas)))
    for ax, p in zip(np.atleast_1d(axes), placas):
        g = m[m["Placa"] == p]
        ax.plot(g["Fecha"], g["Kilometraje acumulado"] / 1000, color=c["gris"], linewidth=1.2, zorder=1)
        gp, gc = g[g["es_prev"]], g[g["es_corr"]]
        ax.scatter(gp["Fecha"], gp["Kilometraje acumulado"] / 1000, s=16, color=c["gris"], edgecolor="white",
                   linewidth=0.6, zorder=2)
        ax.scatter(gc["Fecha"], gc["Kilometraje acumulado"] / 1000, s=46, color=c["acento"], edgecolor="white",
                   linewidth=1, zorder=3)
        ax.set_ylabel(f"{p}\nmil km", fontsize=9)
    np.atleast_1d(axes)[0].set_title("Kilometraje acumulado por camioneta · gris: preventivo · rojo: correctivo")
    fig.tight_layout()
    plt.close(fig)
    return fig


# =====================================================================================================
# 3. EDA DEL DETAILED EVENT REPORT (TELEMETRÍA GPS)
# =====================================================================================================
def convertir_tipos_eventos(raw):
    """Corrige los tipos del Detailed Event Report y crea las columnas de trabajo (trabaja sobre una copia).

    Correcciones:
        - StartOdo y EventValue: texto con coma de miles ("68,784.40") -> número.
        - EventStartDate+EventStartTime -> inicio; EventEndDate+EventEndTime -> fin (formato dd/mm/aaaa hh:mm:ss).
        - TotalDuration (hh:mm:ss) -> duracion_s (segundos).

    Parámetros
    ----------
    raw : DataFrame  (resultado de leer_eventos)

    Devuelve
    -------
    (DataFrame, dict) : eventos convertidos y un resumen de la conversión
        {"no_nulos_antes", "no_nulos_despues", "fechas_inicio_invalidas", "fechas_fin_invalidas",
         "duraciones_invalidas", "rango"}.
    """
    ev = raw.copy()
    antes, despues = {}, {}
    for c in ["StartOdo", "EventValue"]:
        antes[c] = int(ev[c].notna().sum())
        ev[c] = pd.to_numeric(ev[c].astype("string").str.replace(",", "", regex=False), errors="coerce")
        despues[c] = int(ev[c].notna().sum())
    ev["inicio"] = pd.to_datetime(ev["EventStartDate"] + " " + ev["EventStartTime"],
                                  format="%d/%m/%Y %H:%M:%S", errors="coerce")
    ev["fin"] = pd.to_datetime(ev["EventEndDate"] + " " + ev["EventEndTime"],
                               format="%d/%m/%Y %H:%M:%S", errors="coerce")
    ev["duracion_s"] = pd.to_timedelta(ev["TotalDuration"], errors="coerce").dt.total_seconds()
    info = {"no_nulos_antes": antes, "no_nulos_despues": despues,
            "fechas_inicio_invalidas": int(ev["inicio"].isna().sum()),
            "fechas_fin_invalidas": int(ev["fin"].isna().sum()),
            "duraciones_invalidas": int(ev["duracion_s"].isna().sum()),
            "rango": f"{ev['inicio'].min()} → {ev['inicio'].max()}"}
    return ev, info


def valores_centinela(ev):
    """Cuenta valores de texto que no son nulos reales: '#Error', '[UNKNOWN]' y 'N/A'.

    Devuelve
    -------
    DataFrame : columna, n.º de valores centinela y porcentaje sobre el total de filas.
    """
    filas = []
    for col in ev.select_dtypes(include=["string", "object"]).columns:
        n = ev[col].astype("string").str.contains(r"#Error|\[UNKNOWN\]|^N/?A$", regex=True, na=False).sum()
        if n:
            filas.append({"columna": col, "centinelas": int(n), "% de filas": round(n / len(ev) * 100, 1)})
    return pd.DataFrame(filas).set_index("columna") if filas else pd.DataFrame()


def voltaje_en_cero(ev):
    """Cuenta lecturas de voltaje (unidad mv) iguales a 0: lecturas ausentes disfrazadas de cero.

    Devuelve
    -------
    dict : {"lecturas_voltaje": int, "en_cero": int, "% en_cero": float}
    """
    v = ev[ev["MeasurementUnits"] == "mv"]
    z = int((v["EventValue"] == 0).sum())
    return {"lecturas_voltaje": len(v), "en_cero": z, "% en_cero": round(z / max(len(v), 1) * 100, 1)}


def reporte_duplicados_eventos(ev):
    """Duplicados en eventos: filas completas, EventKey repetido y eventos repetidos por camioneta+instante+tipo.

    Devuelve
    -------
    dict : {"filas_completas", "EventKey_repetido", "repetidos_camioneta_inicio_tipo",
            "repetidos_camioneta_inicio_fin_tipo_odo"}
    """
    return {"filas_completas": int(ev.duplicated().sum()),
            "EventKey_repetido": int(ev["EventKey"].duplicated().sum()),
            "repetidos_camioneta_inicio_tipo": int(ev.duplicated(["AssetExtra", "inicio", "EventDescription"], keep=False).sum()),
            "repetidos_camioneta_inicio_fin_tipo_odo": int(ev.duplicated(
                ["AssetExtra", "inicio", "fin", "EventDescription", "StartOdo"], keep=False).sum())}


def cortes_red_gemelos(ev):
    """Cuenta los cortes de red que se registran a la vez como 'No GSM Coverage' y 'No GPRS Coverage'.

    Sirve para no sobrecontar cortes de cobertura: un mismo corte suele aparecer dos veces.

    Devuelve
    -------
    dict : {"cortes_distintos": int, "registrados_como_gsm_y_gprs": int, "eventos_de_red": int}
    """
    red = ev[ev["EventDescription"].str.contains("No G", na=False)]
    g = red.groupby(["AssetExtra", "inicio", "fin"])["EventDescription"].nunique()
    return {"cortes_distintos": len(g), "registrados_como_gsm_y_gprs": int((g == 2).sum()), "eventos_de_red": len(red)}


def _columnas_identificadores():
    """Columnas con identificadores y columnas a revisar con el equipo (uso interno)."""
    ident = {"AssetName": "nombre del activo (contiene la placa real si hay coincidencias)",
             "AssetID": "ID interno del activo en la plataforma GPS",
             "AssetID2": "ID interno del conductor/llave",
             "AssetExtra2": "código numérico del conductor"}
    revisar = {"Asset2SiteName": "nombres de sitios/clientes de los conductores",
               "F_StartStreet / F_EndStreet": "calles de inicio y fin",
               "StartLatLong / EndLatLong": "coordenadas exactas",
               "AssetSiteName / ReportGroup": "nombre del contrato y del grupo de reporte"}
    return ident, revisar


def revisar_confidencialidad(ev):
    """Revisa si el archivo 'anonimizado' aún contiene placas reales o identificadores. NO imprime valores.

    Busca en AssetName un texto con formato de placa colombiana (3 letras + 3 dígitos) y lista las columnas
    con identificadores (que se deben eliminar) y las que conviene revisar con el equipo.

    Devuelve
    -------
    dict : {"filas_con_placa_real": int, "placas_distintas": int,
            "identificadores": DataFrame (columna, valores únicos, descripción),
            "a_revisar_con_el_equipo": DataFrame}
    """
    ident, revisar = _columnas_identificadores()
    placa = ev["AssetName"].astype("string").str.extract(r"^([A-Z]{3}-?\d{3})")[0]
    return {"filas_con_placa_real": int(placa.notna().sum()), "placas_distintas": int(placa.nunique()),
            "identificadores": pd.DataFrame([{"columna": k, "únicos": ev[k].nunique(), "descripción": v}
                                             for k, v in ident.items() if k in ev.columns]).set_index("columna"),
            "a_revisar_con_el_equipo": pd.DataFrame({"descripción": revisar})}


def eliminar_identificadores(ev):
    """Elimina de la copia de trabajo las columnas con identificadores (AssetName, AssetID, AssetID2, AssetExtra2).

    Después de llamarla ya no queda ninguna placa real en memoria, ni en los outputs del notebook.

    Devuelve
    -------
    DataFrame : copia de ev sin esas columnas.
    """
    ident, _ = _columnas_identificadores()
    return ev.drop(columns=[c for c in ident if c in ev.columns])


def equivalencia_placas_mascarada(raw, mostrar_completo=False):
    """Tabla de equivalencia AssetExtra (VH-NN) <-> AssetName, enmascarada por defecto (ej.: N**-***_...).

    Parámetros
    ----------
    raw : DataFrame  (eventos crudos, con AssetName)
    mostrar_completo : bool
        True solo para verlo uno mismo; NO dejar guardado el output en el notebook.

    Devuelve
    -------
    DataFrame
    """
    def enmascarar(s):
        return re.sub(r"([A-Z]{3})(-?)(\d{3})", lambda m: m.group(1)[0] + "**" + m.group(2) + "***", s)

    eq = raw[["AssetExtra", "AssetName"]].drop_duplicates().sort_values("AssetExtra").reset_index(drop=True)
    if not mostrar_completo:
        eq["AssetName"] = eq["AssetName"].map(enmascarar)
    return eq


def cobertura_temporal(ev):
    """Rango observado, días con eventos y huecos por camioneta.

    Devuelve
    -------
    dict : {"cobertura": DataFrame (primer/último evento, eventos, días con eventos, días en rango, %),
            "hueco_max_dias": DataFrame}
    """
    cov = ev.groupby("AssetExtra").agg(primer_evento=("inicio", "min"), ultimo_evento=("inicio", "max"),
                                      eventos=("EventKey", "size"),
                                      dias_con_eventos=("inicio", lambda s: s.dt.normalize().nunique()))
    cov["dias_en_rango"] = (cov["ultimo_evento"].dt.normalize() - cov["primer_evento"].dt.normalize()).dt.days + 1
    cov["% dias con eventos"] = (cov["dias_con_eventos"] / cov["dias_en_rango"] * 100).round(1)

    def hueco(s):
        d = s.dt.normalize().drop_duplicates().sort_values().diff().dt.days
        return int(d.max()) if d.notna().any() else 0

    return {"cobertura": cov, "hueco_max_dias": ev.groupby("AssetExtra")["inicio"].apply(hueco).to_frame("dias")}


def hueco_mas_largo(ev, camioneta):
    """Fechas de inicio y fin del hueco más largo sin eventos de una camioneta.

    Parámetros
    ----------
    camioneta : str  (por ejemplo "VH-08")

    Devuelve
    -------
    dict : {"desde": date, "hasta": date, "dias": int}
    """
    d = ev.loc[ev["AssetExtra"] == camioneta, "inicio"].dt.normalize().drop_duplicates().sort_values()
    gap = d.diff().dt.days
    i = gap.idxmax()
    return {"desde": d.shift(1)[i].date(), "hasta": d[i].date(), "dias": int(gap[i])}


def grafico_eventos_mensuales(ev):
    """Eventos por mes y camioneta (una fila por camioneta, misma escala vertical).

    Septiembre de 2025 y noviembre de 2026 son meses parciales.

    Devuelve
    -------
    matplotlib.figure.Figure
    """
    c = colores()
    mes = ev["inicio"].dt.to_period("M").dt.to_timestamp()
    mens = ev.assign(mes=mes).pivot_table(index="mes", columns="AssetExtra", values="EventKey", aggfunc="count").fillna(0)
    fig, axes = plt.subplots(mens.shape[1], 1, sharex=True, sharey=True, figsize=(10, 1.6 * mens.shape[1]))
    for ax, col in zip(np.atleast_1d(axes), mens.columns):
        ax.bar(mens.index, mens[col], width=22, color=c["gris"])
        ax.set_ylabel(col, fontsize=9)
    np.atleast_1d(axes)[0].set_title("Eventos por mes y camioneta")
    fig.tight_layout()
    plt.close(fig)
    return fig


def catalogo_eventos(ev):
    """Catálogo de tipos de evento: conteo por EventType, porcentaje del total y reparto por camioneta.

    Devuelve
    -------
    dict : {"catalogo": DataFrame, "por_camioneta": DataFrame}
    """
    cat = pd.crosstab(ev["EventDescription"], ev["EventType"], margins=True)
    cat["% del total"] = (cat["All"] / len(ev) * 100).round(1)
    return {"catalogo": cat.sort_values("All", ascending=False),
            "por_camioneta": pd.crosstab(ev["EventDescription"], ev["AssetExtra"], margins=True)}


def _eventos_bateria():
    """Descripciones de los 5 tipos de evento que describen el estado eléctrico (uso interno)."""
    return ["P.D.C Voltaje Bateria Promedio", "DLP: Alerta Voltaje Batería (IGNON)<10VPro<5s",
            "DLP: Alerta Voltaje Bajo Batería (IGNON)<9.5VPro", "DLP: Desconexion Bateria", "OBC unit reset"]


def resumen_eventos_bateria(ev):
    """Aísla los eventos relacionados con la batería y los cuenta por camioneta.

    Incluye: voltaje promedio periódico, alertas < 10 V y < 9,5 V, desconexiones de batería y reinicios de la unidad.

    Devuelve
    -------
    dict : {"eventos": int, "% del total": float, "tabla": DataFrame (tipo de evento x camioneta)}
    """
    tipos = _eventos_bateria()
    b = ev[ev["EventDescription"].isin(tipos)]
    t = pd.crosstab(b["EventDescription"], b["AssetExtra"], margins=True)
    return {"eventos": len(b), "% del total": round(len(b) / len(ev) * 100, 1), "tabla": t.loc[tipos + ["All"]]}


def resumen_duracion(ev):
    """Distribución de la duración de los eventos y eventos atípicos (> 1 h, > 24 h, duración cero).

    Devuelve
    -------
    dict : {"por_evento": DataFrame (describe en segundos), "mayores_1h", "mayores_24h", "duracion_cero",
            "fin_antes_de_inicio", "eventos_mayores_24h": DataFrame}
    """
    return {"por_evento": ev.groupby("EventDescription")["duracion_s"].describe(percentiles=[.5, .99]).round(0),
            "mayores_1h": int((ev["duracion_s"] > 3600).sum()), "mayores_24h": int((ev["duracion_s"] > 86400).sum()),
            "duracion_cero": int((ev["duracion_s"] == 0).sum()),
            "fin_antes_de_inicio": int((ev["fin"] < ev["inicio"]).sum()),
            "eventos_mayores_24h": ev.loc[ev["duracion_s"] > 86400, ["AssetExtra", "inicio", "EventDescription", "duracion_s"]]}


def grafico_duracion(ev):
    """Histograma de la duración de los eventos en escala logarítmica (solo duraciones > 0).

    Devuelve
    -------
    matplotlib.figure.Figure
    """
    c = colores()
    fig, ax = plt.subplots(figsize=(8, 2.8))
    v = ev.loc[ev["duracion_s"] > 0, "duracion_s"]
    ax.hist(np.log10(v), bins=40, color=c["gris"], edgecolor="white")
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(["1 s", "10 s", "100 s", "~17 min", "~3 h", "~28 h"])
    ax.set_xlabel("Duración del evento (escala log)")
    ax.set_ylabel("Eventos")
    ax.set_title("Distribución de la duración de los eventos")
    plt.close(fig)
    return fig


def resumen_ralenti(ev):
    """Ralentí por camioneta: eventos, eventos excesivos, horas totales y horas sin eventos > 24 h.

    Devuelve
    -------
    DataFrame : una fila por camioneta.
    """
    ral = ev[ev["EventDescription"].str.startswith("EC Ralenti")]
    t = ral.groupby("AssetExtra").agg(
        eventos=("EventKey", "size"),
        excesivos=("EventDescription", lambda s: int((s == "EC Ralenti excesivo").sum())),
        horas_totales=("duracion_s", lambda s: round(s.sum() / 3600, 1)),
        horas_sin_atipicos=("duracion_s", lambda s: round(s[s <= 86400].sum() / 3600, 1)))
    t["% excesivos"] = (t["excesivos"] / t["eventos"] * 100).round(1)
    return t


def grafico_ralenti(tabla):
    """Barras horizontales con las horas de ralentí por camioneta (sin eventos > 24 h).

    Parámetros
    ----------
    tabla : DataFrame  (resultado de resumen_ralenti)

    Devuelve
    -------
    matplotlib.figure.Figure
    """
    c = colores()
    fig, ax = plt.subplots(figsize=(8, 2.6))
    h = tabla["horas_sin_atipicos"].sort_values()
    ax.barh(h.index, h.values, color=c["gris"], height=0.6)
    ax.set_xlabel("Horas de ralentí (sin eventos > 24 h)")
    ax.set_title("Horas de ralentí por camioneta")
    ax.grid(axis="y", visible=False)
    plt.close(fig)
    return fig


def _voltaje_promedio(ev):
    """Lecturas periódicas de voltaje promedio, en voltios y sin ceros (uso interno)."""
    p = ev[(ev["EventDescription"] == "P.D.C Voltaje Bateria Promedio") & (ev["MeasurementUnits"] == "mv")].copy()
    p["volt"] = p["EventValue"] / 1000
    return p[p["volt"] > 0]


def resumen_voltaje(ev):
    """Resumen del voltaje de batería: lecturas, ceros, distribución por evento y rangos plausibles.

    Un sistema de 12 V debería leer entre ~11 y ~15 V; los ceros se tratan como faltantes.

    Devuelve
    -------
    dict : {"lecturas", "en_cero", "por_evento": DataFrame (describe en V),
            "rangos_promedio": dict, "lecturas_por_camioneta": DataFrame}
    """
    vol = ev[ev["MeasurementUnits"] == "mv"].copy()
    vol["volt"] = vol["EventValue"] / 1000
    p = _voltaje_promedio(ev)
    return {"lecturas": len(vol), "en_cero": int((vol["EventValue"] == 0).sum()),
            "por_evento": vol.groupby("EventDescription")["volt"].describe().round(2),
            "rangos_promedio": {"< 9 V": int((p["volt"] < 9).sum()), "9–11 V": int(p["volt"].between(9, 11).sum()),
                                "11–15 V": int(p["volt"].between(11, 15).sum()), "> 15 V": int((p["volt"] > 15).sum())},
            "lecturas_por_camioneta": pd.crosstab(vol["AssetExtra"], vol["EventDescription"], margins=True)}


def grafico_voltaje_diario(ev):
    """Voltaje promedio diario de la batería por camioneta (una fila por camioneta); línea roja punteada en 11 V.

    Devuelve
    -------
    matplotlib.figure.Figure
    """
    c = colores()
    p = _voltaje_promedio(ev)
    diario = p.assign(dia=p["inicio"].dt.normalize()).groupby(["AssetExtra", "dia"])["volt"].mean().reset_index()
    cams = sorted(diario["AssetExtra"].unique())
    fig, axes = plt.subplots(len(cams), 1, sharex=True, sharey=True, figsize=(10, 1.7 * len(cams)))
    for ax, cam in zip(np.atleast_1d(axes), cams):
        g = diario[diario["AssetExtra"] == cam]
        ax.plot(g["dia"], g["volt"], color=c["gris"], linewidth=0.9)
        ax.axhline(11, color=c["acento"], linewidth=0.8, linestyle="--")
        ax.set_ylabel(f"{cam}\nV", fontsize=9)
    np.atleast_1d(axes)[0].set_title("Voltaje promedio diario de la batería · línea roja: 11 V")
    fig.tight_layout()
    plt.close(fig)
    return fig


def serie_voltaje_mensual(ev, camioneta):
    """Comportamiento mensual de la batería de UNA camioneta.

    Columnas: lecturas, voltaje medio y mínimo (V, sin ceros), alertas < 10 V, alertas < 9,5 V,
    desconexiones de batería y reinicios de la unidad.

    Parámetros
    ----------
    camioneta : str  (código de telemetría, por ejemplo "VH-07")

    Devuelve
    -------
    DataFrame : índice = mes.
    """
    e = ev[ev["AssetExtra"] == camioneta].copy()
    e["mes"] = e["inicio"].dt.to_period("M")
    p = _voltaje_promedio(e)
    p["mes"] = p["inicio"].dt.to_period("M")
    t = p.groupby("mes")["volt"].agg(lecturas="count", volt_medio="mean", volt_min="min").round(2)
    for nombre, patron in (("alertas_<10V", "<10V"), ("alertas_<9.5V", "<9.5V"),
                           ("desconexiones", "Desconexion"), ("reinicios", "OBC unit reset")):
        t[nombre] = e[e["EventDescription"].str.contains(patron, regex=False, na=False)].groupby("mes").size()
    return t.fillna(0).astype({c: int for c in ["alertas_<10V", "alertas_<9.5V", "desconexiones", "reinicios"]})


def correctivos_bateria(mtto, equivalencia):
    """Correctivos de batería del consolidado, con la placa equivalente en la telemetría.

    Usa la equivalencia de placas inferida por continuidad del odómetro (PROVISIONAL: pendiente de confirmar).

    Parámetros
    ----------
    equivalencia : dict  {placa_hojas: placa_telemetria}, p. ej. preprocesamiento.EQUIVALENCIA_PLACAS

    Devuelve
    -------
    DataFrame : placa_hojas, placa_telemetria, Fecha, Descripción.
    """
    c = mtto[mtto["es_corr"] & mtto["Descripción"].str.contains("BATER", case=False, na=False)]
    out = c[["Placa", "Fecha", "Descripción"]].rename(columns={"Placa": "placa_hojas"})
    out.insert(1, "placa_telemetria", out["placa_hojas"].map(equivalencia))
    return out.reset_index(drop=True)


def grafico_bateria_camioneta(ev, camioneta, fechas_cambio=None):
    """Comportamiento de la batería de una camioneta en el tiempo (dos paneles con el mismo eje de fechas).

    Panel superior: voltaje promedio diario y mínimo diario, con líneas de referencia en 11 V y 10 V.
    Panel inferior: alertas < 10 V por semana (barras) y desconexiones/reinicios por semana (puntos rojos).
    Las líneas verticales rojas marcan los cambios de batería registrados en mantenimiento.

    Parámetros
    ----------
    camioneta : str  (código de telemetría)
    fechas_cambio : list[Timestamp] | None

    Devuelve
    -------
    matplotlib.figure.Figure
    """
    c = colores()
    e = ev[ev["AssetExtra"] == camioneta]
    p = _voltaje_promedio(e)
    d = p.assign(dia=p["inicio"].dt.normalize()).groupby("dia")["volt"].agg(["mean", "min"])
    sem = e.assign(semana=e["inicio"].dt.to_period("W").dt.start_time)
    alert = sem[sem["EventDescription"].str.contains("<10V", regex=False, na=False)].groupby("semana").size()
    elec = sem[sem["EventDescription"].str.contains("Desconexion|OBC unit reset", na=False)].groupby("semana").size()
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(10, 4.6), gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(d.index, d["mean"], color=c["gris"], linewidth=1.1, label="media diaria")
    a1.plot(d.index, d["min"], color=c["tinta"], linewidth=0.6, alpha=0.6, label="mínimo diario")
    a1.axhline(11, color=c["acento"], linewidth=0.8, linestyle="--")
    a1.axhline(10, color=c["acento"], linewidth=0.8, linestyle=":")
    a1.set_ylabel("Voltaje (V)")
    a1.legend(frameon=False, loc="lower left", fontsize=8)
    a1.set_title(f"Batería de {camioneta} en el tiempo · líneas rojas horizontales: 11 V y 10 V")
    a2.bar(alert.index, alert.values, width=5, color=c["gris"], label="alertas < 10 V (semana)")
    if len(elec):
        a2.scatter(elec.index, elec.values, s=24, color=c["acento"], edgecolor="white", linewidth=0.6, zorder=3,
                   label="desconexiones + reinicios (semana)")
    a2.set_ylabel("Eventos por semana")
    a2.legend(frameon=False, loc="upper left", fontsize=8)
    for f in (fechas_cambio or []):
        for a in (a1, a2):
            a.axvline(pd.Timestamp(f), color=c["acento"], linewidth=1.4)
    fig.tight_layout()
    plt.close(fig)
    return fig


def resumen_odometro(ev):
    """Cobertura y coherencia del odómetro (StartOdo > 0), retrocesos y km/día entre días consecutivos.

    Devuelve
    -------
    dict : {"filas_con_odometro", "pct_con_odometro", "por_evento", "por_camioneta", "retrocesos",
            "rango_por_camioneta", "dias_con_lectura", "pares_dias_consecutivos", "km_dia_por_camioneta",
            "dias_km_cero", "dias_mas_500_km", "dias_negativos"}
    """
    tiene = ev["StartOdo"] > 0
    odo = ev[tiene].sort_values(["AssetExtra", "inicio"]).copy()
    odo["d_odo"] = odo.groupby("AssetExtra")["StartOdo"].diff()
    odo["dia"] = odo["inicio"].dt.normalize()
    dmax = odo.groupby(["AssetExtra", "dia"])["StartOdo"].max().reset_index()
    dmax["km_dia"] = dmax.groupby("AssetExtra")["StartOdo"].diff()
    dmax["gap"] = dmax.groupby("AssetExtra")["dia"].diff().dt.days
    cons = dmax[dmax["gap"] == 1]
    fmt = lambda g: g.agg(con_odometro="sum", total="size", pct=lambda s: round(s.mean() * 100, 1))
    return {"filas_con_odometro": int(tiene.sum()), "pct_con_odometro": round(float(tiene.mean() * 100), 1),
            "por_evento": fmt(tiene.groupby(ev["EventDescription"])),
            "por_camioneta": fmt(tiene.groupby(ev["AssetExtra"])),
            "retrocesos": int((odo["d_odo"] < 0).sum()),
            "rango_por_camioneta": odo.groupby("AssetExtra")["StartOdo"].agg(minimo="min", maximo="max", lecturas="count"),
            "dias_con_lectura": len(dmax), "pares_dias_consecutivos": len(cons),
            "km_dia_por_camioneta": cons.groupby("AssetExtra")["km_dia"].describe().round(1),
            "dias_km_cero": int((cons["km_dia"] == 0).sum()), "dias_mas_500_km": int((cons["km_dia"] > 500).sum()),
            "dias_negativos": int((cons["km_dia"] < 0).sum())}


def grafico_odometro(ev):
    """Odómetro por camioneta en el tiempo (solo eventos con lectura), una fila por camioneta.

    Devuelve
    -------
    matplotlib.figure.Figure
    """
    c = colores()
    odo = ev[ev["StartOdo"] > 0].sort_values(["AssetExtra", "inicio"])
    n = odo["AssetExtra"].nunique()
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, 1.7 * n))
    for ax, (cam, g) in zip(np.atleast_1d(axes), odo.groupby("AssetExtra")):
        ax.plot(g["inicio"], g["StartOdo"] / 1000, color=c["gris"], linewidth=1.1)
        ax.set_ylabel(f"{cam}\nmil km", fontsize=9)
    np.atleast_1d(axes)[0].set_title("Odómetro por camioneta (solo eventos con lectura)")
    fig.tight_layout()
    plt.close(fig)
    return fig


def resumen_conductores(ev):
    """Conductores por camioneta y porcentaje de eventos sin conductor identificado.

    Devuelve
    -------
    dict : {"conductores_distintos", "eventos_sin_conductor", "pct_sin_conductor",
            "por_camioneta": DataFrame, "top_conductores": DataFrame}
    """
    sin = ev["AssetName2"].eq("Conductor no identificado")
    por = ev.assign(sin_conductor=sin).groupby("AssetExtra").agg(
        eventos=("EventKey", "size"), conductores=("AssetName2", "nunique"),
        pct_sin_conductor=("sin_conductor", lambda s: round(s.mean() * 100, 1)))
    return {"conductores_distintos": int(ev["AssetName2"].nunique()), "eventos_sin_conductor": int(sin.sum()),
            "pct_sin_conductor": round(float(sin.mean() * 100), 1), "por_camioneta": por,
            "top_conductores": ev["AssetName2"].value_counts().head(10).to_frame("eventos")}


def resumen_ubicacion(ev):
    """Calidad de las coordenadas y de la geocodificación. Solo conteos y rangos agregados (sin mapas).

    Devuelve
    -------
    dict : {"coordenadas": DataFrame (nulos, formato inválido, fuera de rango y rango de lat/lon por columna),
            "ciudad_inicio": DataFrame, "pct_sin_coordenadas_por_camioneta": DataFrame}
    """
    filas, sin_coord = [], None
    for nombre, col in (("inicio", "StartLatLong"), ("fin", "EndLatLong")):
        ll = ev[col].astype("string").str.extract(r"^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$").astype(float)
        fuera = (~ll[0].between(-90, 90) | ~ll[1].between(-180, 180)) & ll[0].notna()
        filas.append({"punto": nombre, "nulos": int(ev[col].isna().sum()),
                      "% nulos": round(ev[col].isna().mean() * 100, 1),
                      "formato_invalido": int((ev[col].notna() & ll[0].isna()).sum()), "fuera_de_rango": int(fuera.sum()),
                      "lat_min": round(ll[0].min(), 2), "lat_max": round(ll[0].max(), 2),
                      "lon_min": round(ll[1].min(), 2), "lon_max": round(ll[1].max(), 2)})
        if nombre == "inicio":
            sin_coord = (ll[0].isna().groupby(ev["AssetExtra"]).mean() * 100).round(1).to_frame("% sin coordenadas")
    return {"coordenadas": pd.DataFrame(filas).set_index("punto"),
            "ciudad_inicio": ev["F_StartCity"].value_counts(dropna=False).to_frame("eventos"),
            "pct_sin_coordenadas_por_camioneta": sin_coord}


def resumen_red(ev, cobertura):
    """Pérdida de cobertura GSM/GPRS: eventos y horas por camioneta, y % del tiempo observado en ese estado.

    La duración incluye el tiempo con el vehículo apagado o estacionado, y cada tipo se mide por separado
    (GSM y GPRS no se suman porque suelen ser el mismo corte).

    Parámetros
    ----------
    cobertura : DataFrame  (clave "cobertura" de cobertura_temporal)

    Devuelve
    -------
    dict : {"eventos_y_horas": DataFrame, "pct_tiempo_sin_cobertura": DataFrame, "eventos_mayores_24h": int}
    """
    red = ev[ev["EventDescription"].str.contains("No G", na=False)]
    ev_h = red.groupby(["AssetExtra", "EventDescription"]).agg(
        eventos=("EventKey", "size"), horas=("duracion_s", lambda s: round(s.sum() / 3600, 1))).unstack()
    horas_obs = (cobertura["dias_en_rango"] * 24).rename("horas_observadas")
    h = red.pivot_table(index="AssetExtra", columns="EventDescription", values="duracion_s", aggfunc="sum").div(3600)
    pct = h.div(horas_obs, axis=0).mul(100).round(1)
    pct.columns = [f"% tiempo {c.split(': ')[-1]}" for c in pct.columns]
    return {"eventos_y_horas": ev_h, "pct_tiempo_sin_cobertura": pd.concat([horas_obs, pct], axis=1),
            "eventos_mayores_24h": int((red["duracion_s"] > 86400).sum())}


def comparar_llaves(ev, mtto, equivalencia=None):
    """Compara las placas de telemetría con las del consolidado y el traslape de fechas entre ambas fuentes.

    Sin `equivalencia`, las placas VH-01..05 (hojas) y VH-06..10 (telemetría) no coinciden. La equivalencia
    inferida por odómetro (preprocesamiento.EQUIVALENCIA_PLACAS) es provisional.

    Parámetros
    ----------
    equivalencia : dict | None  {placa_hojas: placa_telemetria}

    Devuelve
    -------
    dict : {"placas_telemetria", "placas_consolidado", "en_ambos", "con_equivalencia": DataFrame o None,
            "traslape": str, "mantenimientos_en_traslape", "correctivos_en_traslape"}
    """
    p_gps, p_m = sorted(ev["AssetExtra"].dropna().astype(str).unique()), sorted(mtto["Placa"].astype(str).unique())
    ini, fin = max(ev["inicio"].min(), mtto["Fecha"].min()), min(ev["inicio"].max(), mtto["Fecha"].max())
    en = mtto["Fecha"].between(ini, fin)
    eq = None
    if equivalencia:
        eq = pd.DataFrame({"placa_hojas": list(equivalencia), "placa_telemetria": list(equivalencia.values())})
        eq["en_telemetria"] = eq["placa_telemetria"].isin(p_gps)
    return {"placas_telemetria": p_gps, "placas_consolidado": p_m, "en_ambos": sorted(set(p_gps) & set(p_m)),
            "con_equivalencia": eq, "traslape": f"{ini.date()} → {fin.date()} ({(fin - ini).days} días)",
            "mantenimientos_en_traslape": int(en.sum()), "correctivos_en_traslape": int((en & mtto["es_corr"]).sum())}
