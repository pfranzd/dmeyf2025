# src/final_training.py
import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .config import FINAL_TRAIN, FINAL_PREDIC, SEMILLA
from .best_params import cargar_mejores_hiperparametros
from .gain_function import ganancia_lgb_binary

logger = logging.getLogger(__name__)

def preparar_datos_entrenamiento_final(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.
  
    Args:
        df: DataFrame con todos los datos
  
    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDIC}")
  
    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN

    df_train = df[df['foto_mes'].isin(FINAL_TRAIN)]
    df_predict = df[df['foto_mes'] == FINAL_PREDIC]
  
    # Datos de predicción: período FINAL_PREDIC 

    logger.info(f"Registros de entrenamiento: {len(df_train):,}")
    logger.info(f"Registros de predicción: {len(df_predict):,}")
  
    #Corroborar que no esten vacios los df

    # Preparar features y target para entrenamiento
  
    X_train = df_train.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1, errors = 'ignore')
    y_train = df_train['clase_ternaria'].values

    # Preparar features para predicción
    X_predict = df_predict.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1, errors = 'ignore')
    clientes_predict = df_predict['numero_de_cliente'].values

    # logger.info(f"Features utilizadas: {len(features_cols)}")
    logger.info(f"Distribución del target - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")
  
    return X_train, y_train, X_predict, clientes_predict

def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series, mejores_params: dict) -> lgb.Booster:
    """
    Entrena el modelo final con los mejores hiperparámetros.
  
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna
  
    Returns:
        lgb.Booster: Modelo entrenado
    """
    logger.info("Iniciando entrenamiento del modelo final")
  
    # Configurar parámetros del modelo
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'verbose': -1,
        **mejores_params  # Agregar los mejores hiperparámetros
    }
  
    logger.info(f"Parámetros del modelo: {params}")
  
    # Crear dataset de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
  
    # Entrenar modelo con lgb.train()
    modelo = lgb.train(params=params, train_set = train_data, num_boost_round=40)

    return modelo

def generar_predicciones_finales(
    modelo: lgb.Booster,
    X_predict: pd.DataFrame,
    clientes_predict: np.ndarray,
    umbral: float = 0.025,
    tipo_umbral: str = "probabilidad"
) -> pd.DataFrame:
    """
    Genera las predicciones finales para el período objetivo.

    Args:
        modelo (lgb.Booster): Modelo entrenado.
        X_predict (pd.DataFrame): Features para predicción.
        clientes_predict (np.ndarray): IDs de clientes.
        umbral (float | int): Umbral para clasificación.
            - Si tipo_umbral='probabilidad', se interpreta como probabilidad mínima (0-1).
            - Si tipo_umbral='cantidad', se interpreta como cantidad de registros (entero > 0).
        tipo_umbral (str): Tipo de umbral a aplicar ('probabilidad' o 'cantidad').

    Returns:
        pd.DataFrame: DataFrame con columnas ['numero_de_cliente', 'Predicted'].
    """
    logger.info("Generando predicciones finales...")

    # 1️⃣ Predicciones de probabilidad
    y_pred_prob = modelo.predict(X_predict)
    n_clientes = len(y_pred_prob)

    if len(clientes_predict) != n_clientes:
        raise ValueError("La longitud de clientes_predict no coincide con las predicciones generadas.")

    # 2️⃣ Cálculo del umbral según tipo
    if tipo_umbral.lower() == "probabilidad":
        y_pred = (y_pred_prob >= umbral).astype(int)
        logger.info(f"Usando umbral de probabilidad = {umbral:.4f}")

    elif tipo_umbral.lower() == "cantidad":
        if not isinstance(umbral, (int, np.integer)) or umbral <= 0:
            raise ValueError("Para tipo_umbral='cantidad', el umbral debe ser un entero positivo.")

        top_n = min(umbral, n_clientes)
        umbral_prob = np.sort(y_pred_prob)[-top_n]  # prob mínima dentro del top N
        y_pred = (y_pred_prob >= umbral_prob).astype(int)
        logger.info(f"Usando umbral por cantidad: Top {top_n:,} clientes (prob mínima={umbral_prob:.4f})")

    else:
        raise ValueError("El parámetro tipo_umbral debe ser 'probabilidad' o 'cantidad'.")

    # 3️⃣ Construcción del DataFrame final (solo 2 columnas)
    resultados = pd.DataFrame({
        'numero_de_cliente': clientes_predict,
        'Predicted': y_pred
    })

    # 4️⃣ Estadísticas informativas
    total_pred = len(resultados)
    positivos = resultados['Predicted'].sum()
    pct_positivos = positivos / total_pred * 100

    logger.info(f"Resultados de predicción final:")
    logger.info(f"  Total clientes: {total_pred:,}")
    logger.info(f"  Predicciones positivas: {positivos:,} ({pct_positivos:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_pred - positivos:,}")
    logger.info(f"  Tipo de umbral: {tipo_umbral}")
    logger.info(f"  Valor de umbral: {umbral}")

    return resultados

def entrenar_modelos_multiples_semillas(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    mejores_params: dict
) -> list:
    """
    Entrena múltiples modelos de LightGBM usando distintas semillas definidas en SEMILLA.
    Devuelve una lista de modelos entrenados.

    Args:
        X_train (pd.DataFrame): Features de entrenamiento.
        y_train (pd.Series): Target de entrenamiento.
        mejores_params (dict): Mejores hiperparámetros obtenidos con Optuna.

    Returns:
        list[lgb.Booster]: Lista de modelos entrenados con distintas semillas.
    """
    modelos = []

    logger.info(f"Entrenando {len(SEMILLA)} modelos con distintas semillas para ensamblado...")
    for i, seed in enumerate(SEMILLA, start=1):
        logger.info(f"🔁 Entrenando modelo {i}/{len(SEMILLA)} con seed={seed}")

        params = {
            'objective': 'binary',
            'metric': 'None',  # usamos métrica custom (fuera del training)
            'random_state': seed,
            'verbose': -1,
            **mejores_params,
            # parámetros adicionales para reproducibilidad total
            'bagging_seed': seed,
            'feature_fraction_seed': seed,
            'data_random_seed': seed,
            'drop_seed': seed,
            'deterministic': True
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        modelo = lgb.train(params=params, train_set=train_data, num_boost_round=40)

        modelos.append(modelo)
        logger.info(f"✅ Modelo {i} entrenado con éxito (seed={seed})")

    logger.info(f"Entrenamiento múltiple finalizado: {len(modelos)} modelos generados.")
    return modelos


def generar_predicciones_ensamble(
    modelos: list,
    X_predict: pd.DataFrame,
    clientes_predict: np.ndarray,
    umbral: float = 0.025,
    tipo_umbral: str = "probabilidad"
) -> pd.DataFrame:
    """
    Genera predicciones finales promediando las probabilidades de múltiples modelos entrenados con distintas semillas.

    Args:
        modelos (list[lgb.Booster]): Lista de modelos entrenados.
        X_predict (pd.DataFrame): Features de predicción.
        clientes_predict (np.ndarray): IDs de clientes.
        umbral (float | int): Umbral para clasificación.
            - Si tipo_umbral='probabilidad', se interpreta como probabilidad mínima (0-1).
            - Si tipo_umbral='cantidad', se interpreta como cantidad de registros (entero > 0).
        tipo_umbral (str): Tipo de umbral ('probabilidad' o 'cantidad').

    Returns:
        pd.DataFrame: DataFrame con columnas ['numero_de_cliente', 'Predicted'].
    """
    logger.info(f"Generando predicciones con ensamble de {len(modelos)} modelos...")

    # 1️⃣ Generar predicciones de probabilidad para cada modelo
    todas_predicciones = []
    for i, modelo in enumerate(modelos, start=1):
        y_pred_prob = modelo.predict(X_predict)
        todas_predicciones.append(y_pred_prob)
        logger.info(f"Modelo {i}: Probabilidades generadas (min={y_pred_prob.min():.4f}, max={y_pred_prob.max():.4f})")

    # 2️⃣ Promedio de probabilidades (ensembling)
    y_pred_promedio = np.mean(np.column_stack(todas_predicciones), axis=1)
    logger.info("Probabilidades promediadas entre modelos.")

    n_clientes = len(y_pred_promedio)
    if len(clientes_predict) != n_clientes:
        raise ValueError("La longitud de clientes_predict no coincide con las predicciones generadas.")

    # 3️⃣ Cálculo del umbral según tipo
    if tipo_umbral.lower() == "probabilidad":
        y_pred = (y_pred_promedio >= umbral).astype(int)
        logger.info(f"Usando umbral de probabilidad = {umbral:.4f}")

    elif tipo_umbral.lower() == "cantidad":
        if not isinstance(umbral, (int, np.integer)) or umbral <= 0:
            raise ValueError("Para tipo_umbral='cantidad', el umbral debe ser un entero positivo.")

        top_n = min(umbral, n_clientes)
        umbral_prob = np.sort(y_pred_promedio)[-top_n]
        y_pred = (y_pred_promedio >= umbral_prob).astype(int)
        logger.info(f"Usando umbral por cantidad: Top {top_n:,} clientes (prob mínima={umbral_prob:.4f})")

    else:
        raise ValueError("El parámetro tipo_umbral debe ser 'probabilidad' o 'cantidad'.")

    # 4️⃣ Construcción del DataFrame final (igual que la función original)
    resultados = pd.DataFrame({
        'numero_de_cliente': clientes_predict,
        'Predicted': y_pred
    })

    # 5️⃣ Estadísticas informativas
    total_pred = len(resultados)
    positivos = resultados['Predicted'].sum()
    pct_positivos = positivos / total_pred * 100

    logger.info(f"Resultados finales del ensamble:")
    logger.info(f"  Total clientes: {total_pred:,}")
    logger.info(f"  Predicciones positivas: {positivos:,} ({pct_positivos:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_pred - positivos:,}")
    logger.info(f"  Tipo de umbral: {tipo_umbral}")
    logger.info(f"  Valor de umbral: {umbral}")
    logger.info(f"  Modelos combinados: {len(modelos)}")

    return resultados