# src/features.py
import pandas as pd
import duckdb
import logging

logger = logging.getLogger("__name__")

def feature_engineering_business(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la etapa de feature engineering de negocio sobre el DataFrame de entrada.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original con las columnas crudas de entrada sobre las cuales se aplican
        las transformaciones.

    Returns
    -------
    pd.DataFrame
        DataFrame transformado con las nuevas variables de negocio agregadas.
    """

    logger.info(f"Realizando business feature engineering")

    # Construir la consulta SQL
    sql = "SELECT *"
  
    sql += """ EXCLUDE(clase_ternaria),
    (ctarjeta_debito_transacciones + ctarjeta_visa_transacciones + ctarjeta_master_transacciones) as ctarjeta_total_transacciones,
    (ctarjeta_visa_transacciones + ctarjeta_master_transacciones) as ctarjeta_credito_transacciones,
    if(cpayroll_trx > 0 , 1, 0) as cpayroll,
    (mcaja_ahorro + mcaja_ahorro_adicional + mcaja_ahorro_dolares) as mcaja_ahorro_total,
    (mtarjeta_visa_consumo + mtarjeta_master_consumo) as mtarjeta_credito_consumo,
    (mplazo_fijo_pesos + mplazo_fijo_dolares) as mplazo_fijo_total,
    (Visa_mlimitecompra + Master_mlimitecompra) as credito_mlimitecompra,
    (mcuenta_corriente_adicional+mcuenta_corriente+mcaja_ahorro+mcaja_ahorro_adicional+mcaja_ahorro_dolares+mcuentas_saldo+mplazo_fijo_dolares+mplazo_fijo_pesos+minversion1_pesos+minversion1_dolares+minversion2) as mbanco_total,
    (mprestamos_personales + mprestamos_prendarios + mprestamos_hipotecarios) as mprestamos_total,
    if(mrentabilidad_annual = 0, 0, mrentabilidad / mrentabilidad_annual) as porc_mrentabilidad_annual,
    clase_ternaria"""
  
    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    print(df.head())
  
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df

def feature_engineering_business_alternative(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering de negocio: genera variables derivadas de relaciones
    entre las columnas originales (sin usar lags ni medias móviles).
    """

    logger.info("Iniciando business feature engineering (versión corregida sin columnas inexistentes)")

    sql = """
    SELECT 
        * EXCLUDE(clase_ternaria),

        -- TARJETAS Y TRANSACCIONES
        (ctarjeta_debito_transacciones + ctarjeta_visa_transacciones + ctarjeta_master_transacciones) AS ctarjeta_total_transacciones,
        (ctarjeta_visa_transacciones + ctarjeta_master_transacciones) AS ctarjeta_credito_transacciones,
        (mtarjeta_visa_consumo + mtarjeta_master_consumo) AS mtarjeta_credito_consumo,
        (Visa_mlimitecompra + Master_mlimitecompra) AS credito_mlimitecompra,
        (mtarjeta_credito_consumo / NULLIF(credito_mlimitecompra, 0)) AS ratio_consumo_credito,
        IF(mtarjeta_credito_consumo > 0, 1, 0) AS usa_tarjeta_credito,

        -- CUENTAS Y DEPÓSITOS (estos son pasivos desde el punto de vista del banco)
        (mcaja_ahorro + mcaja_ahorro_adicional + mcaja_ahorro_dolares) AS mcaja_ahorro_total,
        (mcuenta_corriente + mcuenta_corriente_adicional) AS mcuenta_corriente_total,
        (mplazo_fijo_pesos + mplazo_fijo_dolares) AS mplazo_fijo_total,
        (mcaja_ahorro_total + mcuenta_corriente_total + mplazo_fijo_total) AS mpasivos_total,
        (mcaja_ahorro_total / NULLIF(mpasivos_total, 0)) AS ratio_ahorro_cuentas,
        IF(mpasivos_total > 0, 1, 0) AS usa_cuentas,

        -- INVERSIONES (activos del cliente)
        (minversion1_pesos + minversion1_dolares + minversion2) AS minversiones_total,
        IF(minversiones_total > 0, 1, 0) AS usa_inversiones,

        -- PRÉSTAMOS
        (mprestamos_personales + mprestamos_prendarios + mprestamos_hipotecarios) AS mprestamos_total,
        IF(mprestamos_total > 0, 1, 0) AS usa_prestamos,

        -- ACTIVOS Y TOTALES DEL CLIENTE
        (minversiones_total + mprestamos_total) AS mactivos_total,
        (mactivos_total + mpasivos_total) AS mbanco_total,

        -- MÁRGENES
        (mpasivos_margen + mactivos_margen) AS margen_total,
        (margen_total / NULLIF(mactivos_total, 0)) AS ratio_margen_activos,
        (margen_total / NULLIF(mpasivos_total, 0)) AS ratio_margen_pasivos,
        (margen_total / NULLIF(mbanco_total, 0)) AS ratio_margen_banco,

        -- RATIOS FINANCIEROS
        (mactivos_total / NULLIF(mpasivos_total, 0)) AS ratio_activo_pasivo,

        -- RENTABILIDAD
        CASE 
            WHEN mrentabilidad_annual = 0 THEN 0 
            ELSE mrentabilidad / mrentabilidad_annual 
        END AS porc_mrentabilidad_annual,

        -- ENGAGEMENT
        IF(cpayroll_trx > 0, 1, 0) AS tiene_payroll,
        (ctrx_quarter / 3.0) AS ctrx_promedio_mensual,
        (ctarjeta_total_transacciones / NULLIF(ctrx_quarter, 0)) AS ratio_tarjeta_transacciones,

        -- DIVERSIFICACIÓN
        ((usa_tarjeta_credito + usa_inversiones + usa_prestamos + tiene_payroll + usa_cuentas) / 5.0) AS ratio_uso_productos,

        clase_ternaria
    FROM df
    """

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df


def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """


    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
  
    # Construir la consulta SQL
    sql = "SELECT *"
  
    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")
  
    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    print(df.head())
  
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df

def feature_engineering_moving_avg(df: pd.DataFrame, columnas: list[str], cant_ventanas: int = 3) -> pd.DataFrame:
    """
    Genera variables de medias móviles (rolling means) para los atributos especificados utilizando SQL (DuckDB).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos.
    columnas : list[str]
        Lista de atributos para los cuales generar medias móviles. Si es None o vacío, no se generan.
    cant_ventanas : int, default=3
        Cantidad de períodos (meses) a considerar en la ventana móvil.

    Returns
    -------
    pd.DataFrame
        DataFrame con las variables de medias móviles agregadas.
    """

    logger.info(f"Realizando feature engineering de medias móviles con ventana de {cant_ventanas} meses "
                f"para {len(columnas) if columnas else 0} atributos")

    columnas = [col for col in columnas if "_lag_" not in col]

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar medias móviles")
        return df

    # Construir la consulta SQL base
    sql = "SELECT *"

    # Agregar las medias móviles para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += (
                f", avg({attr}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes "
                f"ROWS BETWEEN {cant_ventanas - 1} PRECEDING AND CURRENT ROW) AS {attr}_moving_avg_{cant_ventanas}"
            )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    print(df.head())

    logger.info(f"Feature engineering de medias móviles completado. DataFrame resultante con {df.shape[1]} columnas")

    return df