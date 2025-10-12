# src/optimizacion_cv.py
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
import os
import logging
from .config import (
    SEMILLA, MES_TRAIN, MES_VALIDACION, STUDY_NAME,
    GANANCIA_ACIERTO, COSTO_ESTIMULO, PARAMETROS_LGB
    # , FEATURES
)
from .gain_function import ganancia_evaluator
from src.tp.grafico_test import generar_grafico_importancia, generar_graficos_optuna

logger = logging.getLogger(__name__)

# Función objetivo para Optuna con Cross Validation temporal (respetando estructura temporal)
def objetivo_ganancia_temporal(trial, df) -> float:
    """
    Función objetivo para Optuna con Cross Validation temporal.
    Evalúa la ganancia promedio sobre folds secuenciales de meses,
    respetando la estructura temporal de los datos.

    Args:
        trial: Trial de Optuna
        df: DataFrame con datos (con columna 'foto_mes' y 'clase_ternaria')

    Returns:
        float: Ganancia promedio del CV temporal
    """

    # ================================
    # 1️⃣ Hiperparámetros a optimizar
    # ================================
    params = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'num_leaves': trial.suggest_int('num_leaves', PARAMETROS_LGB['num_leaves'][0], PARAMETROS_LGB['num_leaves'][1]),
        'learning_rate': trial.suggest_float('learning_rate', PARAMETROS_LGB['learning_rate'][0], PARAMETROS_LGB['learning_rate'][1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', PARAMETROS_LGB['feature_fraction'][0], PARAMETROS_LGB['feature_fraction'][1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', PARAMETROS_LGB['bagging_fraction'][0], PARAMETROS_LGB['bagging_fraction'][1]),
        'min_child_samples': trial.suggest_int('min_child_samples', PARAMETROS_LGB['min_child_samples'][0], PARAMETROS_LGB['min_child_samples'][1]),
        'max_depth': trial.suggest_int('max_depth', PARAMETROS_LGB['max_depth'][0], PARAMETROS_LGB['max_depth'][1]),
        'reg_alpha': trial.suggest_float('reg_alpha', PARAMETROS_LGB['reg_alpha'][0], PARAMETROS_LGB['reg_alpha'][1]),
        'reg_lambda': trial.suggest_float('reg_lambda', PARAMETROS_LGB['reg_lambda'][0], PARAMETROS_LGB['reg_lambda'][1]),
        'bin': trial.suggest_int('bin', PARAMETROS_LGB['bin'][0], PARAMETROS_LGB['bin'][1]),
        'verbosity': -1,
        'random_state': SEMILLA[0]
    }

    # ================================
    # 2️⃣ Definición de folds temporales
    # ================================
    # ⚠️ Asegúrate de que estos meses existan en tu dataset
    folds_temporales = [
        ([202101], 202102),
        ([202101, 202102], 202103),
        ([202101, 202102, 202103], 202104)
    ]

    features = [col for col in df.columns if col not in ["clase_ternaria"]]
    target = "clase_ternaria"
    ganancias = []

    # ================================
    # 3️⃣ Loop temporal (sin leakage)
    # ================================
    
    for i, (train_meses, valid_mes) in enumerate(folds_temporales, 1):
        df_train = df[df["foto_mes"].isin(train_meses)]
        df_valid = df[df["foto_mes"] == valid_mes]

        X_train, y_train = df_train[features], df_train[target]
        X_valid, y_valid = df_valid[features], df_valid[target]

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid)

        modelo = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_valid],
            feval=ganancia_evaluator,
            num_boost_round=150,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # Predicciones y cálculo de ganancia del fold
        y_pred = modelo.predict(X_valid)
        ganancia_fold = ganancia_evaluator(y_pred, lgb_valid)[1]
        ganancias.append(ganancia_fold)

        logger.info(f"Fold {i}: Train={train_meses}, Valid={valid_mes}, Ganancia={ganancia_fold:,.0f}")

    # ================================
    # 4️⃣ Resultado promedio del CV
    # ================================
    ganancia_promedio = np.mean(ganancias)
    ganancia_std = np.std(ganancias)

    logger.info(f"Trial {trial.number} - Ganancia promedio CV temporal: {ganancia_promedio:,.0f} ± {ganancia_std:,.0f}")

    # ================================
    # 5️⃣ Guardar resultados del trial
    # ================================
    resultados_dir = "optuna_results"
    os.makedirs(resultados_dir, exist_ok=True)
    registro = {
        "trial": trial.number,
        "ganancia_promedio": float(ganancia_promedio),
        "ganancia_std": float(ganancia_std),
        "params": params
    }

    with open(os.path.join(resultados_dir, f"cv_temporal_trial_{trial.number}.json"), "w") as f:
        json.dump(registro, f, indent=4)

    return ganancia_promedio

def objetivo_ganancia_cv(trial, df) -> float:
    """
    Función objetivo para Optuna con Cross Validation.
    Utiliza SEMILLA[0] desde configuración para reproducibilidad.

    Args:
        trial: Trial de Optuna
        df: DataFrame con datos
  
    Returns:
        float: Ganancia promedio del CV
    """
    # Hiperparámetros a optimizar (desde configuración YAML)
    params = {
        'objective': 'binary',
        'metric': 'custom',  # Usamos nuestra métrica personalizada
        'num_leaves': trial.suggest_int('num_leaves', PARAMETROS_LGB['num_leaves'][0], PARAMETROS_LGB['num_leaves'][1]),
        'learning_rate': trial.suggest_float('learning_rate', PARAMETROS_LGB['learning_rate'][0], PARAMETROS_LGB['learning_rate'][1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', PARAMETROS_LGB['feature_fraction'][0], PARAMETROS_LGB['feature_fraction'][1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', PARAMETROS_LGB['bagging_fraction'][0], PARAMETROS_LGB['bagging_fraction'][1]),
        'min_child_samples': trial.suggest_int('min_child_samples', PARAMETROS_LGB['min_child_samples'][0], PARAMETROS_LGB['min_child_samples'][1]),
        'max_depth': trial.suggest_int('max_depth', PARAMETROS_LGB['max_depth'][0], PARAMETROS_LGB['max_depth'][1]),
        'reg_alpha': trial.suggest_float('reg_alpha', PARAMETROS_LGB['reg_alpha'][0], PARAMETROS_LGB['reg_alpha'][1]),
        'reg_lambda': trial.suggest_float('reg_lambda', PARAMETROS_LGB['reg_lambda'][0], PARAMETROS_LGB['reg_lambda'][1]),
        'bin': trial.suggest_int('bin', PARAMETROS_LGB['bin'][0], PARAMETROS_LGB['bin'][1]),
        'random_state': SEMILLA[0],  # Desde configuración YAML
        'verbosity': -1
    }
  
    # Preparar datos para CV

    try:
        df_train = df[df["foto_mes"].isin(MES_TRAIN)]
        df_valid = df[df["foto_mes"] == MES_VALIDACION]
    except KeyError as e:
        raise KeyError(f"Falta columna esperada en el DataFrame: {e}")
   
  
    # Features y target
    features = [col for col in df.columns if col not in ["clase_ternaria"]]
    target = "clase_ternaria"

    X = df_train[features]
    y = df_train[target]

    # Crear dataset de LightGBM

    train_data = lgb.Dataset(X, label=y, free_raw_data=False)
  
    # Configurar CV con semilla desde configuración
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=150,
        nfold=5,
        stratified=True,
        feval=ganancia_evaluator,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        seed = SEMILLA[0]
    )
  
    # Extraer ganancia promedio y max
  
    metric_key = list(cv_results.keys())[0]  # ej. 'ganancia-mean'
    ganancia_promedio = np.max(cv_results[metric_key])
    ganancia_std = cv_results[metric_key.replace('mean', 'stdv')][np.argmax(cv_results[metric_key])]

    # Guardar iteración para análisis posterior

    resultados_dir = "optuna_results"
    os.makedirs(resultados_dir, exist_ok=True)
    registro = {
        "trial": trial.number,
        "ganancia_promedio": float(ganancia_promedio),
        "ganancia_std": float(ganancia_std),
        "params": params
    }
  
    # Agregar nueva iteración
  
    with open(os.path.join(resultados_dir, f"cv_trial_{trial.number}.json"), "w") as f:
        json.dump(registro, f, indent=4)

    logger.info(f"Iteración CV {trial.number} - Ganancia: {ganancia_promedio:,.0f} ± {ganancia_std:,.0f}")

    return ganancia_promedio


def optimizar_con_cv(df, n_trials=50) -> optuna.Study:
    """
    Ejecuta optimización bayesiana con Cross Validation.
  
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
  
    Returns:
        optuna.Study: Estudio de Optuna con resultados de CV
    """
    study_name = f"{STUDY_NAME}"
  
    logger.info(f"Iniciando optimización con CV - {n_trials} trials")
    logger.info(f"Configuración CV: períodos={MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]}")
  
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )
  
    # Ejecutar optimización
    # study.optimize(lambda trial: objetivo_ganancia_cv(trial, df), n_trials=n_trials)
    study.optimize(lambda trial: objetivo_ganancia_temporal(trial, df), n_trials=n_trials)

    # Obtener los mejores parámetros
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': SEMILLA[0]
    })

    # Preparar dataset final
    df_train = df[df["foto_mes"].isin(MES_TRAIN)]
    X_train = df_train[[col for col in df.columns if col not in ["clase_ternaria"]]]
    y_train = df_train["clase_ternaria"]

    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

    # Recalcular best_round usando CV final (con best_params)
    cv_results_final = lgb.cv(
        best_params,
        train_data,
        num_boost_round=150,
        nfold=5,
        stratified=True,
        feval=ganancia_evaluator,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        seed=SEMILLA[0]
    )

    best_round = len(cv_results_final[list(cv_results_final.keys())[0]])
    print(f"Best round final: {best_round}")

    # Entrenar modelo final con los mejores hiperparámetros y best_round
    model_final = lgb.train(best_params, train_data, num_boost_round=best_round)

    # Generar gráfico de importancia una sola vez
    generar_grafico_importancia(model_final, titulo_estudio="Estudio Churn FY26 - Modelo Óptimo")

    # ===========================
    # GUARDAR RESULTADOS FINALES
    # ===========================
    resultados = []
    for t in study.trials:
        resultados.append({
            "trial_number": t.number,
            "value": t.value,
            "params": t.params
        })

    os.makedirs("resultados", exist_ok=True)
    archivo_final = f"resultados/{study_name}_iteraciones.json"

    with open(archivo_final, "w") as f:
        json.dump(resultados, f, indent=4)

    logger.info(f"Resultados completos guardados en {archivo_final}")
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    try:
        generar_graficos_optuna(study, study_name)
    except Exception as e:
        logger.warning(f"No se pudieron generar los gráficos de Optuna: {e}")

    # Resultados
    return study
