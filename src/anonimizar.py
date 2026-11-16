"""
Pipeline de anonimización (Guía 4).

Entrada : data/raw/            (ignorada por Git; datos reales de la Compañía)
Salidas : data/anonimizados/   (lo único que se publica)
          data/llaves/equivalencias.json  (ignorada por Git; NUNCA se publica)
          docs/log_anonimizacion.csv      (solo conteos, sin valores reales)

Aplica la tabla de equivalencias acordada por el equipo:
- Placas            -> VH-01, VH-02... (un solo código por placa en TODAS las fuentes)
- Conductores       -> Conductor 01, 02...; cédulas y códigos internos se eliminan
- Talleres          -> Taller 01, 02...
- Órdenes de servicio -> OS-001, OS-002... (se conserva la repetición para no ocultar errores de calidad)
- Contratos/campos  -> Contrato A-H; empresa -> la Compañía
- Coordenadas, calles, direcciones, geocercas -> se eliminan; ciudad/municipio -> región
- Documentos del vehículo (motor, VIN, chasis, licencia, SOAT, pólizas, avalúo) -> se eliminan
- Fechas, km, costos, voltajes y eventos -> se mantienen

Cada fuente tiene una política EXPLÍCITA por columna (conservar, codificar, eliminar
o generalizar). Una columna que no esté en la política se elimina por defecto, para
que un cambio en el formato de origen nunca filtre información.

Al final se valida cada archivo de salida: si aparece una placa, un nombre de
conductor, una cédula, un nombre de campo o de empresa real, el archivo NO se exporta.
"""

from pathlib import Path
import json
import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "anonimizados"
KEYS_DIR = BASE_DIR / "data" / "llaves"
DOCS_DIR = BASE_DIR / "docs"
KEYS_FILE = KEYS_DIR / "equivalencias.json"

ARCHIVO_CONSOLIDADO = "Consolidado_Hojas_de_Vida.xlsx"
ARCHIVO_EVENTOS = "Detailed Event Report.csv"
ARCHIVO_VIAJES = "Daily Movement Report.csv"

PATRON_PLACA = re.compile(r"(?<![A-Z0-9])([A-Z]{3})-?(\d{3})(?![0-9])")
PATRON_PLACA_I = re.compile(PATRON_PLACA.pattern, re.IGNORECASE)

# ---------------------------------------------------------------------------
# Equivalencias de texto (tabla del equipo; valores fuera del repositorio)
# ---------------------------------------------------------------------------
# Las equivalencias de nombres reales (contratos, campos, municipios, empresas) NO están en
# este código, porque el repositorio es público y revelarían a qué corresponde cada código.
# Se leen de data/llaves/config_anonimizacion.json (ignorado por Git), que entrega el
# integrante vinculado a la Compañía. Estructura:
#   {"SITIOS": [[patrón_regex, "Contrato X (Región)"], ...],
#    "REGIONES": {"municipio o departamento": "Región", ...},
#    "TERMINOS_PROHIBIDOS": ["término real que nunca debe aparecer", ...]}
CONFIG_FILE = KEYS_DIR / "config_anonimizacion.json"
if not CONFIG_FILE.exists():
    raise SystemExit(f"Falta {CONFIG_FILE.relative_to(BASE_DIR)} (no se versiona). Pídalo al enlace con la Compañía.")
_CONFIG = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
SITIOS = [tuple(x) for x in _CONFIG["SITIOS"]]
REGIONES = _CONFIG["REGIONES"]
TERMINOS_PROHIBIDOS = _CONFIG["TERMINOS_PROHIBIDOS"]

# ---------------------------------------------------------------------------
# Políticas por fuente: columna -> acción
#   "conservar" | "eliminar" | "placa" | "conductor" | "taller" | "orden"
#   | "sitio" | "region" | ("texto", ) -> texto libre con reemplazo de términos
# Columnas no listadas se eliminan.
# ---------------------------------------------------------------------------
POLITICA_VEHICULOS = {
    "Placa": "placa", "Marca": "conservar", "Línea": "conservar",
    "Modelo": "conservar", "Clase": "conservar", "Carrocería": "conservar",
    "Color": "conservar", "Combustible": "conservar",
    "Capacidad (kg)": "conservar", "Capacidad (pasajeros)": "conservar",
    "Fecha expedición de placa": "conservar",
    "Conductor / asignación": "texto",
    # Equipamiento de seguridad (Sí/No)
    "ABS": "conservar", "Control de estabilidad (ESP)": "conservar",
    "Neumáticos": "conservar", "Iluminación eficiente": "conservar",
    "Suspensión": "conservar", "Dirección hidráulica": "conservar",
    "Climatización completa": "conservar",
    "Mantenimiento correctivo": "conservar", "Airbag": "conservar",
    "Cinturón de seguridad": "conservar", "Apoyacabezas": "conservar",
    "Elementos de protección personal": "conservar",
    "Carrocería (seguridad)": "conservar", "Chasis (seguridad)": "conservar",
    "Bluetooth incorporado": "conservar", "Mantenimiento preventivo": "conservar",
    "Vigencia SOAT": "conservar",
    "Tanque largo": "conservar", "Tanque ancho": "conservar",
    "Tanque alto": "conservar", "Vencimiento recarga extintor": "conservar",
    "Revisión técnico-mecánica N°": "eliminar",
    # Identificadores del vehículo, documentos, seguros, propietario, ubicación
    "Licencia de tránsito": "eliminar", "N° motor": "eliminar",
    "N° serie": "eliminar", "N° chasis": "eliminar",
    "Propietario": "eliminar", "Empresa afiliada": "eliminar",
    "Factura de venta vehículo": "eliminar", "Factura tanque": "eliminar",
    "N° SOAT": "eliminar", "Seguro todo riesgo - aseguradora": "eliminar",
    "Seguro todo riesgo - vigencia": "eliminar", "Responsabilidad": "eliminar",
    "Deducible": "eliminar", "N° póliza": "eliminar",
    "Vigencia póliza": "eliminar", "Cupo o póliza N°": "eliminar",
    "Impuesto municipal": "eliminar", "Avalúo (COP)": "eliminar",
}

POLITICA_MANTENIMIENTOS = {
    "Placa": "placa", "Fecha": "conservar", "Descripción": "texto",
    "Tipo": "conservar", "Kilometraje acumulado": "conservar",
    "Orden de servicio": "orden", "Costo (COP)": "conservar",
    "Prestador del servicio": "taller",
}

POLITICA_PLAN = {
    "Placa": "placa", "Año del plan": "conservar", "Rutina": "conservar",
    "Servicio del plan (km)": "conservar", "Km programado": "conservar",
    "Fecha programada": "conservar", "Km ejecutado": "conservar",
    "Fecha ejecutada": "conservar", "Texto original programado": "texto",
    "Texto original ejecutado": "texto", "Observación": "texto",
}

POLITICA_EVENTOS = {
    "AssetExtra": "placa",
    "Asset2SiteName": "sitio",
    "AssetName2": "conductor",
    "EventKey": "conservar", "EventDescription": "conservar",
    "EventType": "conservar", "EventStartDate": "conservar",
    "EventStartTime": "conservar", "EventEndDate": "conservar",
    "EventEndTime": "conservar", "TimeZone": "conservar",
    "TotalOccurs": "conservar", "EventValue": "conservar",
    "MeasurementUnits": "conservar", "TotalDuration": "conservar",
    "StartOdo": "conservar", "EndOdo": "conservar",
    "FuelUsed": "conservar", "DistanceCalculatedFromOdo": "conservar",
    "F_StartTown": "region", "F_EndTown": "region",
    # Vacías o constantes, sin información identificable: se conservan para que la
    # limpieza documente su descarte.
    "Asset2ReportGroup": "conservar", "FleetNumber": "conservar",
    "TripID": "conservar", "RoadSpeedLimit": "conservar",
    "EventLocation": "conservar", "F_StartRegion": "conservar",
    "F_EndRegion": "conservar",
    "AssetSiteName": "sitio", "ReportGroup": "texto",
    # Se eliminan: placa en texto (AssetName, ReportGroup), IDs internos
    # (AssetID, AssetID2, AssetExtra2), sitio del activo (constante),
    # calles, barrios, ciudades, geocercas y coordenadas.
    "AssetName": "eliminar", "AssetID": "eliminar",
    "AssetExtra2": "eliminar", "AssetID2": "eliminar",
    "F_StartStreet": "eliminar", "F_StartSuburb": "eliminar",
    "F_StartCity": "eliminar", "F_EndStreet": "eliminar",
    "F_EndSuburb": "eliminar", "F_EndCity": "eliminar",
    "StartLocation": "eliminar", "EndLocation": "eliminar",
    "StartLatLong": "eliminar", "EndLatLong": "eliminar",
}

POLITICA_VIAJES = {
    "AssetExtra": "placa",
    "SiteName": "sitio",
    "AssetName2": "conductor",
    "DepartureDate": "conservar", "DepartureTime": "conservar",
    "TimeZone1": "conservar", "TripType": "conservar",
    "DrivingTime": "conservar", "StandingTime": "conservar",
    "Duration": "conservar", "Distance": "conservar",
    "StartOdoMeter": "conservar", "EndOdoMeter": "conservar",
    "MaxSpeed": "conservar", "AvgSpeed": "conservar",
    "ArrivalTime": "conservar", "ArrivalDate": "conservar",
    "TimeZone2": "conservar", "NextDepartureDateTime": "conservar",
    "TimeZone": "conservar", "TimeAtLocation": "conservar",
    "NrOfVisits": "conservar", "TotalFuelUsedMeasured": "conservar",
    "CalcFuelConsumption": "conservar",
    "DepartFrom": "region", "ArriveAt": "region",
    "FleetNumber": "conservar", "TripClassificationComment": "conservar",
    "PassengerList": "conservar", "BlankJourneyType": "conservar",
    "AssetHostID": "eliminar", "AssetID": "eliminar",
    "AssetExtra2": "eliminar", "StartLatLong": "eliminar",
    "EndLatLong": "eliminar",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sin_tildes(texto):
    t = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in t if not unicodedata.combining(c)).lower().strip()


def normalizar_placa(valor):
    """'ABC-123_Marca_Linea_2024', 'ABC 123', 'abc123' -> 'ABC123'."""
    if pd.isna(valor):
        return None
    m = PATRON_PLACA.search(str(valor).upper())
    return f"{m.group(1)}{m.group(2)}" if m else None


def normalizar_taller(valor):
    """Unifica variantes de escritura del mismo taller antes de codificar.
    'TALLERES AUTORIZADOS NISSAN', 'TALLER AUTORIZADO NISSAN' y
    'TALLERES AUTORIZADOS' son el mismo prestador (red autorizada de la marca)."""
    t = sin_tildes(valor)
    t = re.sub(r"\btalleres\b", "taller", t)
    t = re.sub(r"\bautorizados\b", "autorizado", t)
    t = re.sub(r"\s+nissan\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


def separar_conductor(valor):
    """'NOMBRE APELLIDO_<cédula>_<empresa>' -> ('nombre apellido', ['<cédula>'])."""
    if pd.isna(valor):
        return None, []
    partes = str(valor).split("_")
    nombre = sin_tildes(partes[0])
    numeros = [p for p in partes[1:] if p.isdigit()]
    return nombre, numeros


def cargar_llaves():
    if KEYS_FILE.exists():
        return json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    return {}


def guardar_llaves(llaves):
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    KEYS_FILE.write_text(json.dumps(llaves, ensure_ascii=False, indent=2), encoding="utf-8")


def asignar_codigos(valores, existentes, prefijo, ancho, separador=" "):
    """Conserva los códigos ya asignados y agrega los nuevos en orden alfabético,
    de modo que volver a correr el pipeline no cambia los códigos."""
    codigos = dict(existentes)
    usados = len(codigos)
    for v in sorted(set(valores) - set(codigos)):
        usados += 1
        codigos[v] = f"{prefijo}{separador}{usados:0{ancho}d}"
    return codigos


def generalizar_region(valor):
    if pd.isna(valor):
        return valor
    t = sin_tildes(valor)
    if "unknown" in t or t == "":
        return "Sin dato"
    for clave, region in REGIONES.items():
        if re.search(rf"\b{re.escape(clave)}\b", t):
            return region
    return "Otra región"


def generalizar_sitio(valor):
    if pd.isna(valor):
        return valor
    t = sin_tildes(valor)
    for patron, nombre in SITIOS:
        if re.search(patron, t):
            return nombre
    return "Otro grupo"


def reemplazar_texto(valor, llaves):
    """Texto libre: reemplaza placas y nombres de empresas/campos."""
    if pd.isna(valor):
        return valor
    texto = PATRON_PLACA_I.sub(
        lambda m: llaves["placas"].get(f"{m.group(1).upper()}{m.group(2)}", "VH-??"),
        str(valor),
    )
    for patron, nombre in SITIOS:
        texto = re.sub(patron, nombre, texto, flags=re.IGNORECASE)
    for patron, nombre in _CONFIG.get("EMPRESAS", []):
        texto = re.sub(patron, nombre, texto, flags=re.IGNORECASE)
    return texto


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def cargar_fuentes():
    consolidado = pd.read_excel(RAW_DIR / ARCHIVO_CONSOLIDADO, sheet_name=None, dtype={"Orden de servicio": str})
    leer = lambda nombre: pd.read_csv(RAW_DIR / nombre, encoding="utf-8-sig", dtype=str)
    fuentes = {
        "consolidado": consolidado,
        "eventos": leer(ARCHIVO_EVENTOS),
        "viajes": leer(ARCHIVO_VIAJES) if (RAW_DIR / ARCHIVO_VIAJES).exists() else None,
    }
    return fuentes


def construir_llaves(fuentes, llaves):
    """Recorre TODAS las fuentes antes de anonimizar, para que cada placa,
    conductor, taller y orden tenga un único código en todos los archivos."""
    placas, conductores, talleres, ordenes, cedulas = set(), set(), set(), set(), set()

    c = fuentes["consolidado"]
    for hoja in c.values():
        if "Placa" in hoja:
            placas |= {p for p in hoja["Placa"].map(normalizar_placa) if p}
    m = c["Mantenimientos"]
    talleres |= set(m["Prestador del servicio"].dropna().map(normalizar_taller))
    ordenes |= set(m["Orden de servicio"].dropna().astype(str).str.strip())

    for nombre in ("eventos", "viajes"):
        df = fuentes[nombre]
        if df is None:
            continue
        # La placa aparece en varias columnas de texto (AssetName, ReportGroup...)
        for col in df.columns:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                muestra = df[col].dropna().astype(str).str.upper().unique()
                for v in muestra:
                    for m_ in PATRON_PLACA.finditer(v):
                        placas.add(f"{m_.group(1)}{m_.group(2)}")
        for v in df["AssetName2"].dropna().unique():
            nombre_c, nums = separar_conductor(v)
            conductores.add(nombre_c)
            cedulas.update(nums)
        for col in ("AssetExtra2", "AssetID2"):
            if col in df:
                cedulas.update(df[col].dropna().astype(str).str.replace(r"\.0$", "", regex=True))

    # Las placas de la flota analizada (hoja Vehículos) reciben los primeros códigos.
    flota = sorted({normalizar_placa(p) for p in c["Vehículos"]["Placa"]})
    llaves["placas"] = asignar_codigos(flota, llaves.get("placas", {}), "VH-", 2, separador="")
    llaves["placas"] = asignar_codigos(placas, llaves["placas"], "VH-", 2, separador="")
    desconocido = {n for n in conductores if "desconocido" in n}
    llaves["conductores"] = asignar_codigos(conductores - desconocido, llaves.get("conductores", {}), "Conductor", 2)
    for n in desconocido:
        llaves["conductores"][n] = "Conductor sin identificar"
    llaves["talleres"] = asignar_codigos(talleres, llaves.get("talleres", {}), "Taller", 2)
    llaves["ordenes"] = asignar_codigos(ordenes, llaves.get("ordenes", {}), "OS-", 3, separador="")
    llaves["_cedulas"] = sorted(cedulas)  # solo para validar; nunca se exporta
    return llaves


# ---------------------------------------------------------------------------
# Anonimización
# ---------------------------------------------------------------------------
def aplicar_politica(df, politica, llaves, nombre, log):
    salida = pd.DataFrame(index=df.index)
    for col in df.columns:
        accion = politica.get(col, "eliminar")
        if accion == "eliminar":
            continue
        s = df[col]
        if accion == "conservar":
            salida[col] = s
        elif accion == "placa":
            salida[col] = s.map(lambda v: llaves["placas"].get(normalizar_placa(v)) if pd.notna(v) else v)
        elif accion == "conductor":
            salida[col] = s.map(lambda v: llaves["conductores"].get(separar_conductor(v)[0]) if pd.notna(v) else v)
        elif accion == "taller":
            salida[col] = s.map(lambda v: llaves["talleres"].get(normalizar_taller(v)) if pd.notna(v) else v)
        elif accion == "orden":
            salida[col] = s.map(lambda v: llaves["ordenes"].get(str(v).strip()) if pd.notna(v) else v)
        elif accion == "sitio":
            salida[col] = s.map(generalizar_sitio)
        elif accion == "region":
            salida[col] = s.map(generalizar_region)
        elif accion == "texto":
            salida[col] = s.map(lambda v: reemplazar_texto(v, llaves))
        log.append({"archivo": nombre, "columna": col, "accion": accion})
    for col in df.columns:
        if politica.get(col, "eliminar") == "eliminar":
            motivo = "política" if col in politica else "no listada (eliminada por defecto)"
            log.append({"archivo": nombre, "columna": col, "accion": f"eliminar ({motivo})"})
    return salida


def agregar_formato_orden(original, anonimizado):
    """El código OS-NNN oculta el número real; antes de reemplazarlo se deja una
    bandera de si el número original seguía el patrón de 8 dígitos (6905xxxx o 6900xxxx).
    Así la limpieza puede reportar este problema de calidad sin conocer el número."""
    orden = original["Orden de servicio"].astype(str).str.strip()
    anonimizado["Orden de servicio formato válido"] = orden.str.fullmatch(r"690\d{5}")
    return anonimizado


# ---------------------------------------------------------------------------
# Validación: si encuentra algo, el archivo NO se exporta
# ---------------------------------------------------------------------------
def validar(df, nombre, llaves):
    nombres_conductores = [n for n in llaves["conductores"]
                           if "desconocido" not in n and re.search(r"[a-z]{3}", n)]
    cedulas = {c for c in llaves["_cedulas"] if len(c) >= 6}
    patron_placa = PATRON_PLACA.pattern.replace("([A-Z]{3})", "(?:[A-Z]{3})").replace(r"(\d{3})", r"(?:\d{3})")
    hallazgos = []
    for col in df.columns:
        texto = df[col].dropna().astype(str)
        if texto.empty:
            continue
        unicos = pd.Series(texto.unique())
        bajo = unicos.map(sin_tildes)
        if unicos.str.upper().str.contains(patron_placa, regex=True).any():
            hallazgos.append((col, "placa real"))
        for termino in TERMINOS_PROHIBIDOS:
            if bajo.str.contains(re.escape(termino)).any():
                hallazgos.append((col, f"término '{termino}'"))
        for n in nombres_conductores:
            if bajo.str.contains(re.escape(n)).any():
                hallazgos.append((col, "nombre de conductor"))
                break
        if unicos.str.replace(r"\.0$", "", regex=True).isin(cedulas).any():
            hallazgos.append((col, "cédula o código interno"))
    if hallazgos:
        print(f"  ✗ {nombre}: NO se exporta. Hallazgos:")
        for col, tipo in sorted(set(hallazgos)):
            print(f"      - columna '{col}': {tipo}")
    return not hallazgos


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    fuentes = cargar_fuentes()
    llaves = construir_llaves(fuentes, cargar_llaves())
    log = []
    resultado_ok = True

    # Consolidado (3 hojas)
    c = fuentes["consolidado"]
    hojas = {
        "Vehículos": aplicar_politica(c["Vehículos"], POLITICA_VEHICULOS, llaves, "Consolidado/Vehículos", log),
        "Mantenimientos": agregar_formato_orden(
            c["Mantenimientos"],
            aplicar_politica(c["Mantenimientos"], POLITICA_MANTENIMIENTOS, llaves, "Consolidado/Mantenimientos", log),
        ),
        "Plan de mantenimiento": aplicar_politica(c["Plan de mantenimiento"], POLITICA_PLAN, llaves, "Consolidado/Plan", log),
    }
    if all(validar(h, f"Consolidado/{n}", llaves) for n, h in hojas.items()):
        with pd.ExcelWriter(OUT_DIR / "Consolidado_Hojas_de_Vida_anonimizado.xlsx") as w:
            for n, h in hojas.items():
                h.to_excel(w, sheet_name=n, index=False)
        print("  ✓ Consolidado_Hojas_de_Vida_anonimizado.xlsx")
    else:
        resultado_ok = False

    # Eventos y viajes
    for clave, politica, salida in (
        ("eventos", POLITICA_EVENTOS, "Detailed Event Report_anonimizado.csv"),
        ("viajes", POLITICA_VIAJES, "Daily Movement Report_anonimizado.csv"),
    ):
        df = fuentes[clave]
        if df is None:
            print(f"  - {clave}: no se encontró el archivo de origen")
            continue
        anon = aplicar_politica(df, politica, llaves, clave, log)
        if validar(anon, clave, llaves):
            anon.to_csv(OUT_DIR / salida, index=False, encoding="utf-8-sig")
            print(f"  ✓ {salida} ({len(anon):,} filas, {anon.shape[1]} columnas)")
        else:
            resultado_ok = False

    guardar_llaves({k: v for k, v in llaves.items() if not k.startswith("_")})
    pd.DataFrame(log).to_csv(DOCS_DIR / "log_anonimizacion.csv", index=False, encoding="utf-8-sig")
    resumen = {k: len(v) for k, v in llaves.items() if not k.startswith("_")}
    print(f"Códigos asignados: {resumen}")
    print("Log de columnas: docs/log_anonimizacion.csv")
    if not resultado_ok:
        raise SystemExit("Hay archivos que no pasaron la validación; revise los hallazgos.")


if __name__ == "__main__":
    print("PIPELINE DE ANONIMIZACIÓN")
    main()
