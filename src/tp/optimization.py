# src/optimization.py
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary

logger = logging.getLogger(__name__)

def objetivo_ganancia(trial, df) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """

    # Definir rangos de hiperparámetros a optimizar
    num_leaves = trial.suggest_int('num_leaves', 8, 100)
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3) # mas bajo, más iteraciones necesita
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 1000)
    feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0)
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 1.0)

    # Hiperparámetros a optimizar
    params = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'min_data_in_leaf': min_data_in_leaf,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'random_state': SEMILLA[0],  # Desde configuración YAML
        'verbose': -1
    }


    # Completar!!!!!!

    ganancia_total = calcular_ganancia(y_val, y_pred_binary)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)
  
    logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    return ganancia_total


def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION
        }
    }
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")


def optimizar(df, n_trials=100) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study = STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    
    # Completar!!!!!!!!
  
    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
  
  
    return study

# src/optimization.py (implementación simplificada)

def evaluar_en_test(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]

    X_train = df_train_completo.drop(columns=['clase_ternaria'])
    y_train = df_train_completo['clase_ternaria'].values

    X_test = df_test.drop(columns=['clase_ternaria'])
    y_test = df_test['clase_ternaria'].values

    # Crear datasets de LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    # Predecir en test
    modelo = lgb.train(
        mejores_params, 
        dtrain,
        # valid_sets=[dtrain, dtest],
        # valid_names=['train', 'test'],
        # callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        feval = ganancia_lgb_binary,
        num_boost_round=200,
        )
    y_pred_proba = modelo.predict(X_test)
    y_pred_binary = (y_pred_proba > 0.025).astype(int)

    # Entrenar modelo con mejores parámetros

    # ... Implementar entrenamiento y test con la logica de entrenamiento FINAL para mayor detalle
    # recordar realizar todos los df necesarios y utilizar lgb.train()
  
    # Calcular solo la ganancia
    ganancia_test = calcular_ganancia(y_test, y_pred_binary)
  
    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas)
    }
  
    return resultados, y_pred_proba

def guardar_resultados_test(resultados_test, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.
    """
    # Guarda en resultados/{STUDY_NAME}_test_results.json
    # ... Implementar utilizando la misma logica que cuando guardamos una iteracion de la Bayesiana