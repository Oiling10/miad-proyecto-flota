
from pathlib import Path
import json
import re
import unicodedata

import pandas as pd


#RUTAS Y CONFIGURACIÓN GENERAL

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "anonimizados"
KEYS_DIR = BASE_DIR / "data" / "llaves"

KEYS_FILE = KEYS_DIR / "equivalencias.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYS_DIR.mkdir(parents=True, exist_ok=True)

#Si quedan posibles datos confidenciales sin resolver, no se exporta el archivo.
BLOQUEAR_SI_HAY_ALERTAS = True

#Columnas cuyos valores deben conservarse sin modificaciones.
COLUMNAS_CONSERVAR = {
    "startlatlong",
    "endlatlong",
    "coordenadas",
    "coordenadas gps",
    "gps coordinates",
}

#Columnas que se deben eliminar por contener identificadores personales, vehiculares, documentales o información sensible.
COLUMNAS_ELIMINAR = {
    "licencia de transito",
    "n motor",
    "numero de motor",
    "n serie",
    "numero de serie",
    "n chasis",
    "numero de chasis",
    "vin",
    "n soat",
    "numero soat",
    "seguro todo riesgo aseguradora",
    "n poliza",
    "numero poliza",
    "cupo o poliza n",
    "avaluo cop",
}

#Nombres alternativos de columnas que contienen placas.
COLUMNAS_PLACAS = {
    "placa",
    "placas",
    "assetextra",
    "matricula",
    "matricula vehiculo",
    "vehicle plate",
    "license plate",
}

#Nombres alternativos de columnas que contienen conductores u otras personas identificables.
COLUMNAS_PERSONAS = {
    "conductor",
    "nombre conductor",
    "conductor asignado",
    "conductor asignacion",
    "driver",
    "driver name",
    "assetname2",
    "nombre del responsable",
    "responsable",
    "nombre del empleado",
    "nombre del tecnico",
}

# Valores de asignación que no contienen información personal y deben conservarse sin modificaciones.
VALORES_NO_PERSONALES = {
    "asignado a campo",
    "sin asignacion",
    "no aplica",
    "no asignado",
    "disponible",
    "vacante",
    "sin conductor",
}

#Equivalencias de nombres de empresas, clientes, contratos y campos conocidos por el proyecto.
EQUIVALENCIAS_TEXTO = {
    "OMIA COLOMBIA S.A.S.": "La Compañía",
    "OMIA COLOMBIA": "La Compañía",
    "OMIA": "La Compañía",

    "Ecopetrol": "Operadora estatal",
    "Frontera": "Operadora privada 1",
    "TPL": "Operadora privada 2",
    "Promisol": "Operadora privada 3",

    "Campo Piedemonte": "Contrato D (Llanos Orientales)",
    "Piedemonte": "Contrato D (Llanos Orientales)",
    "Turpial": "Contrato A (Magdalena Medio)",
    "Abanico": "Contrato B (Tolima)",
    "Gigante": "Contrato C (Huila)",
    "Livianos": "Contrato E (Llanos Orientales)",
    "Pesados": "Contrato F (Llanos Orientales)",
    "Bonga Mamey": "Contrato G (Caribe)",

    "Mix Telematics": "Plataforma de seguimiento GPS",
    "Syscaf": "Plataforma de seguimiento GPS",
}

#Ubicaciones que deben generalizarse.
EQUIVALENCIAS_UBICACIONES = {
    "Tauramena": "Llanos Orientales",
    "Casanare": "Llanos Orientales",
    "Puerto Boyacá": "Magdalena Medio",
    "Tolima": "Tolima",
    "Huila": "Huila",
    "Meta": "Llanos Orientales",
    "Sucre": "Caribe",
}

#Columnas que sus valores deben convertirse en códigos.
PREFIJOS_IDENTIFICADORES = {
    "placas": ("VH", 2),
    "conductores": ("Conductor", 2),
    "talleres": ("Taller", 2),
    "proveedores": ("Proveedor", 2),
    "ordenes": ("OS", 3),
    "facturas": ("FAC", 3),
}


#FUNCIONES AUXILIARES

def normalizar_texto(valor):
    """Normaliza textos para comparar nombres de columnas."""

    texto = str(valor).strip().lower()

    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caracter for caracter in texto
        if not unicodedata.combining(caracter)
    )

    texto = re.sub(r"[°º]", "", texto)
    texto = re.sub(r"[^a-z0-9]+", " ", texto)

    return re.sub(r"\s+", " ", texto).strip()


def es_valor_vacio(valor):
    """Comprueba si un valor está vacío o es nulo."""

    if pd.isna(valor):
        return True

    return str(valor).strip() == ""


def cargar_equivalencias():
    """Carga las equivalencias existentes o crea los diccionarios."""

    if KEYS_FILE.exists():
        with open(KEYS_FILE, "r", encoding="utf-8") as archivo:
            equivalencias = json.load(archivo)
    else:
        equivalencias = {}

    #Permite incorporar nuevas categorías sin perder las anteriores.
    for categoria in PREFIJOS_IDENTIFICADORES:
        equivalencias.setdefault(categoria, {})

    return equivalencias


def guardar_equivalencias(equivalencias):
    """Guarda las equivalencias localmente."""

    with open(KEYS_FILE, "w", encoding="utf-8") as archivo:
        json.dump(
            equivalencias,
            archivo,
            ensure_ascii=False,
            indent=4
        )


def obtener_codigo(categoria, valor, equivalencias):
    """
    Asigna un código consecutivo y conserva la equivalencia
    en futuras ejecuciones.
    """

    if es_valor_vacio(valor):
        return valor

    valor_original = str(valor).strip()
    registro = equivalencias[categoria]

    #Evita duplicados por diferencias de mayúsculas, minúsculas o espacios.
    for original, codigo in registro.items():
        if normalizar_texto(original) == normalizar_texto(valor_original):
            return codigo

    prefijo, digitos = PREFIJOS_IDENTIFICADORES[categoria]

    numero = len(registro) + 1

    if categoria in {"conductores", "talleres", "proveedores"}:
        codigo = f"{prefijo} {numero:0{digitos}d}"
    else:
        codigo = f"{prefijo}-{numero:0{digitos}d}"

    registro[valor_original] = codigo

    return codigo


#CARGA Y GUARDADO DE ARCHIVOS

def cargar_archivo(ruta):
    """Carga archivos CSV o Excel."""

    extension = ruta.suffix.lower()

    if extension == ".csv":

        ultimo_error = None

        #Probar distintas codificaciones.
        for encoding in ["utf-8-sig", "cp1252", "latin-1"]:
            try:
                return {
                    "Datos": pd.read_csv(
                        ruta,
                        sep=None,
                        engine="python",
                        encoding=encoding
                    )
                }

            except (UnicodeDecodeError, pd.errors.ParserError) as error:
                ultimo_error = error

        raise ValueError(
            f"No fue posible leer el CSV {ruta.name}: {ultimo_error}"
        )

    elif extension in [".xlsx", ".xls"]:

        try:
            return pd.read_excel(ruta, sheet_name=None)

        except ImportError as error:
            raise ImportError(
                "Falta instalar el motor de lectura de Excel. Para archivos XLSX instalar openpyxl y para XLS instalar xlrd."
            ) from error

    else:
        raise ValueError(f"Formato no compatible: {extension}")


def guardar_archivo(hojas, ruta):
    """Guarda los datos anonimizados en CSV o Excel."""

    if ruta.suffix.lower() == ".csv":

        hojas["Datos"].to_csv(
            ruta,
            index=False,
            encoding="utf-8-sig"
        )

    elif ruta.suffix.lower() == ".xlsx":

        with pd.ExcelWriter(ruta, engine="openpyxl") as writer:

            for nombre_hoja, df in hojas.items():

                nombre_seguro = str(nombre_hoja)[:31]

                df.to_excel(
                    writer,
                    sheet_name=nombre_seguro,
                    index=False
                )

    else:
        raise ValueError(f"Formato no compatible: {ruta.suffix}")


#IDENTIFICACIÓN DE VALORES CONFIDENCIALES

def identificar_categoria(nombre_columna):
    """Identifica la categoría de información según el nombre de columna."""

    nombre = normalizar_texto(nombre_columna)

    if nombre in COLUMNAS_PLACAS:
        return "placas"

    if nombre in COLUMNAS_PERSONAS:
        return "conductores"

    if "prestador" in nombre or "taller" in nombre:
        return "talleres"

    if "proveedor" in nombre:
        return "proveedores"

    if "orden de servicio" in nombre or "solped" in nombre:
        return "ordenes"

    if "factura" in nombre:
        return "facturas"

    return None


def recopilar_valores(archivos):
    """Reúne los valores sensibles de todos los archivos y hojas para mantener los códigos consistentes."""

    valores = {
        categoria: set()
        for categoria in PREFIJOS_IDENTIFICADORES
    }

    patron_placa = r"\b[A-Z]{3}[- ]?\d{3}\b"

    for hojas in archivos.values():

        for df in hojas.values():

            for columna in df.columns:

                categoria = identificar_categoria(columna)

                if categoria:

                    for valor in df[columna].dropna():

                        texto = str(valor).strip()

                        if not texto:
                            continue

                        if categoria == "conductores" and (
                            normalizar_texto(texto) in VALORES_NO_PERSONALES
                            or normalizar_texto(texto) in {
                                "conductor desconocido",
                                "desconocido",
                                "no identificado",
                                "conductor no identificado",
                            }
                        ):
                            continue

                        if categoria in {"ordenes", "facturas"}:
                            if normalizar_texto(texto) in {
                                "na", "n a", "no aplica"
                            }:
                                continue

                        valores[categoria].add(texto)

            # Buscar placas dentro de textos, por ejemplo AssetName.
            for columna in df.select_dtypes(include=["object"]).columns:

                if normalizar_texto(columna) in COLUMNAS_CONSERVAR:
                    continue

                for valor in df[columna].dropna():

                    coincidencias = re.findall(
                        patron_placa,
                        str(valor),
                        flags=re.IGNORECASE
                    )

                    for placa in coincidencias:
                        valores["placas"].add(placa.strip())

    return valores


def construir_equivalencias(valores, equivalencias):
    """Crea los códigos anónimos que todavía no existen."""

    for categoria, lista_valores in valores.items():

        for valor in sorted(lista_valores):

            obtener_codigo(
                categoria,
                valor,
                equivalencias
            )


#REEMPLAZO DE INFORMACIÓN EN TEXTOS LIBRES

def reemplazar_en_texto(texto, equivalencias_texto):
    """Reemplaza expresiones conocidas dentro de un texto."""

    if not isinstance(texto, str):
        return texto

    resultado = texto

    #Reemplazar primero las expresiones más largas.
    for original, anonimizado in sorted(
        equivalencias_texto.items(),
        key=lambda elemento: len(elemento[0]),
        reverse=True
    ):

        patron = (
            r"(?<!\w)"
            + re.escape(original)
            + r"(?!\w)"
        )

        resultado = re.sub(
            patron,
            lambda coincidencia: anonimizado,
            resultado,
            flags=re.IGNORECASE
        )

    return resultado


def reemplazar_identificadores(texto, categoria, equivalencias):
    """Reemplaza identificadores conocidos dentro de textos libres."""

    if not isinstance(texto, str):
        return texto

    resultado = texto

    for original, codigo in sorted(
        equivalencias[categoria].items(),
        key=lambda elemento: len(elemento[0]),
        reverse=True
    ):

        patron = (
            r"(?<!\w)"
            + re.escape(original)
            + r"(?!\w)"
        )

        resultado = re.sub(
            patron,
            lambda coincidencia: codigo,
            resultado,
            flags=re.IGNORECASE
        )

    return resultado


def anonimizar_texto(valor, equivalencias):

    if not isinstance(valor, str):
        return valor

    # Conservar valores operativos que no contienen información personal.
    if normalizar_texto(valor) in VALORES_NO_PERSONALES:
        return valor

    texto = valor

    #Empresas, contratos y ubicaciones.
    texto = reemplazar_en_texto(
        texto,
        EQUIVALENCIAS_TEXTO
    )

    texto = reemplazar_en_texto(
        texto,
        EQUIVALENCIAS_UBICACIONES
    )

    #Identificadores conocidos.
    for categoria in PREFIJOS_IDENTIFICADORES:

        texto = reemplazar_identificadores(
            texto,
            categoria,
            equivalencias
        )

    #Placas que puedan aparecer dentro de textos nuevos.
    patron_placa = r"\b[A-Z]{3}[- ]?\d{3}\b"

    coincidencias = re.findall(
        patron_placa,
        texto,
        flags=re.IGNORECASE
    )

    for placa in set(coincidencias):

        codigo = obtener_codigo(
            "placas",
            placa,
            equivalencias
        )

        texto = re.sub(
            re.escape(placa),
            codigo,
            texto,
            flags=re.IGNORECASE
        )

    return texto


#ANONIMIZACIÓN DE COLUMNAS

def anonimizar_valor_columna(valor, categoria, equivalencias):
    """Anonimiza los valores de una columna identificada."""

    if es_valor_vacio(valor):
        return valor

    texto = str(valor).strip()
    texto_normalizado = normalizar_texto(texto)

    # Conservar valores que no identifican personas.
    if categoria == "conductores":

        if texto_normalizado in VALORES_NO_PERSONALES:
            return valor

        if texto_normalizado in {
            "conductor desconocido",
            "desconocido",
            "no identificado",
            "conductor no identificado",
        }:
            return "Conductor no identificado"

    if categoria in {"ordenes", "facturas"}:
        if texto_normalizado in {
            "na", "n a", "no aplica"
        }:
            return valor

    return obtener_codigo(
        categoria,
        texto,
        equivalencias
    )


def eliminar_columnas(df):
    """Elimina las columnas configuradas como confidenciales."""

    columnas_eliminar = []

    for columna in df.columns:

        nombre = normalizar_texto(columna)

        if nombre in {
            normalizar_texto(c)
            for c in COLUMNAS_ELIMINAR
        }:
            columnas_eliminar.append(columna)

    return df.drop(columns=columnas_eliminar)


def anonimizar_hoja(df, equivalencias):
    """Aplica las reglas generales a cualquier hoja de datos."""

    df = df.copy()

    #Eliminar columnas confidenciales.
    df = eliminar_columnas(df)

    for columna in df.columns:

        nombre = normalizar_texto(columna)

        #Conservar las coordenadas GPS sin modificaciones.
        if nombre in {
            normalizar_texto(c)
            for c in COLUMNAS_CONSERVAR
        }:
            continue

        categoria = identificar_categoria(columna)

        if categoria:

            df[columna] = df[columna].apply(
                lambda valor: anonimizar_valor_columna(
                    valor,
                    categoria,
                    equivalencias
                )
            )

        # Aplicar reemplazos en todas las columnas de texto para detectar información identificable en descripciones, observaciones y otros campos libres.
        if (
            pd.api.types.is_object_dtype(df[columna])
            or pd.api.types.is_string_dtype(df[columna])
        ):

            df[columna] = df[columna].apply(
                lambda valor: anonimizar_texto(
                    valor,
                    equivalencias
                )
            )

    return df


#VALIDACIÓN DE POSIBLES DATOS CONFIDENCIALES

def validar_hoja(df, nombre_hoja, equivalencias):
    """Busca patrones conocidos y valores originales que puedan haber quedado sin anonimizar."""

    hallazgos = []

    patrones = {
        "Posible placa sin anonimizar": r"\b[A-Z]{3}[- ]?\d{3}\b",

        "Posible identificación etiquetada": (
            r"\b(?:CC|C\.C\.|NIT|CEDULA|CÉDULA)"
            r"\s*[:#-]?\s*\d{5,}\b"
        ),
    }

    valores_sensibles = []

    for categoria in PREFIJOS_IDENTIFICADORES:

        valores_sensibles.extend(
            equivalencias[categoria].keys()
        )

    valores_sensibles.extend(EQUIVALENCIAS_TEXTO.keys())
    valores_sensibles.extend(EQUIVALENCIAS_UBICACIONES.keys())

    for columna in df.columns:

        nombre = normalizar_texto(columna)

        if nombre in {
            normalizar_texto(c)
            for c in COLUMNAS_CONSERVAR
        }:
            continue

        for indice, valor in df[columna].items():

            if not isinstance(valor, str):
                continue

            #Buscar patrones conocidos.
            for descripcion, patron in patrones.items():

                if re.search(
                    patron,
                    valor,
                    flags=re.IGNORECASE
                ):

                    hallazgos.append({
                        "hoja": nombre_hoja,
                        "fila": indice + 2,
                        "columna": columna,
                        "tipo": descripcion,
                        "valor": valor[:100],
                    })

                    break

            else:

                #Buscar valores originales conocidos.
                for original in valores_sensibles:

                    patron = (
                        r"(?<!\w)"
                        + re.escape(original)
                        + r"(?!\w)"
                    )

                    if re.search(
                        patron,
                        valor,
                        flags=re.IGNORECASE
                    ):

                        hallazgos.append({
                            "hoja": nombre_hoja,
                            "fila": indice + 2,
                            "columna": columna,
                            "tipo": "Posible valor original pendiente",
                            "valor": valor[:100],
                        })

                        break

    return hallazgos


def validar_archivo(hojas, nombre_archivo, equivalencias):
    """Valida todas las hojas antes de exportar."""

    hallazgos = []

    for nombre_hoja, df in hojas.items():

        hallazgos.extend(
            validar_hoja(
                df,
                nombre_hoja,
                equivalencias
            )
        )

    if hallazgos:

        print(
            f"\nADVERTENCIA: se encontraron posibles datos "
            f"pendientes en {nombre_archivo}."
        )

        for hallazgo in hallazgos[:20]:

            print(
                f" - Hoja: {hallazgo['hoja']} | "
                f"Fila: {hallazgo['fila']} | "
                f"Columna: {hallazgo['columna']} | "
                f"{hallazgo['tipo']} | "
                f"Valor: {hallazgo['valor']}"
            )

        if len(hallazgos) > 20:
            print(
                f" - Y {len(hallazgos) - 20} hallazgos adicionales."
            )

        return False

    print(f"Validación automática superada: {nombre_archivo}")

    return True


#PROCESAMIENTO DE ARCHIVOS

def procesar_archivo(archivo, equivalencias):
    """Carga, anonimiza, valida y guarda el archivo."""

    print(f"\nProcesando: {archivo.name}")

    try:

        hojas = cargar_archivo(archivo)

        hojas_anonimizadas = {}

        #Anonimizar todas las hojas.
        for nombre_hoja, df in hojas.items():

            print(f"  Hoja: {nombre_hoja}")

            df_anonimizado = anonimizar_hoja(
                df,
                equivalencias
            )

            hojas_anonimizadas[nombre_hoja] = df_anonimizado

            print(
                f"    Filas: {df_anonimizado.shape[0]} | "
                f"Columnas: {df_anonimizado.shape[1]}"
            )

        #Guardar las equivalencias generadas.
        guardar_equivalencias(equivalencias)

        #Validar antes de exportar.
        if BLOQUEAR_SI_HAY_ALERTAS:

            validacion = validar_archivo(
                hojas_anonimizadas,
                archivo.name,
                equivalencias
            )

            if not validacion:
                print(
                    "Archivo no exportado. "
                    "Requiere revisión manual."
                )
                return

        #Mantener el formato original cuando sea posible.
        if archivo.suffix.lower() == ".xls":

            nombre_salida = (
                archivo.stem + "_anonimizado.xlsx"
            )

        else:

            nombre_salida = (
                archivo.stem
                + "_anonimizado"
                + archivo.suffix
            )

        ruta_salida = OUTPUT_DIR / nombre_salida

        guardar_archivo(
            hojas_anonimizadas,
            ruta_salida
        )

        print(f"Archivo guardado: {ruta_salida.name}")

    except Exception as error:

        print(
            f"Error al procesar {archivo.name}: {error}"
        )


#EJECUCIÓN PRINCIPAL

def procesar_archivos():
    """Procesa todos los archivos compatibles de data/raw."""

    if not RAW_DIR.exists():

        print("No existe la carpeta data/raw/.")
        return

    archivos = [
        archivo
        for archivo in RAW_DIR.iterdir()
        if archivo.is_file()
        and archivo.suffix.lower() in {
            ".csv",
            ".xlsx",
            ".xls"
        }
        and not archivo.name.startswith("~$")
    ]

    if not archivos:

        print("No se encontraron archivos compatibles en data/raw/.")
        return

    print(f"Archivos encontrados: {len(archivos)}")

    #Cargar equivalencias existentes.
    equivalencias = cargar_equivalencias()

    #Cargar todos los archivos para identificar los valores antes de anonimizar para mantener la consistencia de códigos.
    archivos_cargados = {}

    for archivo in archivos:

        try:
            archivos_cargados[archivo] = cargar_archivo(archivo)

        except Exception as error:
            print(f"No se pudo cargar {archivo.name}: {error}")

    if not archivos_cargados:

        print("No fue posible cargar los archivos.")
        return

    #Recopilar valores confidenciales de todos los archivos.
    valores = recopilar_valores(archivos_cargados)

    construir_equivalencias(
        valores,
        equivalencias
    )

    #Guardar equivalencias antes del procesamiento.
    guardar_equivalencias(equivalencias)

    #Procesar cada archivo.
    for archivo, hojas in archivos_cargados.items():

        procesar_archivo(
            archivo,
            equivalencias
        )

    guardar_equivalencias(equivalencias)

    print("\nPIPELINE DE ANONIMIZACIÓN FINALIZADO.")


if __name__ == "__main__":

    print("INICIO DEL PIPELINE DE ANONIMIZACIÓN")

    procesar_archivos()
