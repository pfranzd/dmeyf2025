import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
import os
import logging
from .config import (
    SEMILLA, MES_TRAIN, STUDY_NAME,
    GANANCIA_ACIERTO, COSTO_ESTIMULO, PARAMETROS_LGB
)
from .gain_function import ganancia_evaluator
from src.tp.grafico_test import generar_grafico_importancia, generar_graficos_optuna

logger = logging.getLogger(__name__)

# =====================================
# 🎯 Función objetivo única (CV tradicional)
# =====================================
def objetivo_ganancia_cv(trial, df) -> float:
    """
    Función objetivo para Optuna con Cross Validation estándar (K-fold) de LightGBM.
    Evalúa la ganancia promedio del modelo y guarda resultados por trial.
    """

    # =========================
    # 1️⃣ Definir hiperparámetros
    # =========================
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
        'max_bin': trial.suggest_int('max_bin', PARAMETROS_LGB['max_bin'][0], PARAMETROS_LGB['max_bin'][1]),
        'verbosity': -1,
        # 🔐 Semillas y determinismo para reproducibilidad
        'deterministic': True,
        'bagging_seed': SEMILLA[0],
        'feature_fraction_seed': SEMILLA[0],
        'drop_seed': SEMILLA[0],
        'extra_seed': SEMILLA[0],
        'data_random_seed': SEMILLA[0],
        'num_threads': 1
    }

    # =========================
    # 2️⃣ Filtrar datos y preparar dataset
    # =========================
    df_train = df[df["foto_mes"].isin(MES_TRAIN)]
    features = [c for c in df_train.columns if c != "clase_ternaria"]
    target = "clase_ternaria"

    lgb_train = lgb.Dataset(df_train[features], label=df_train[target], free_raw_data=False)

    # =========================
    # 3️⃣ Cross Validation de LightGBM
    # =========================
    cv_results = lgb.cv(
        params,
        lgb_train,
        num_boost_round=150,
        nfold=5,
        stratified=True,
        feval=ganancia_evaluator,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        seed=SEMILLA[0]
    )

    # Extraer ganancia máxima y desviación estándar
    metric_key = list(cv_results.keys())[0]
    best_idx = np.argmax(cv_results[metric_key])
    ganancia_promedio = cv_results[metric_key][best_idx]
    ganancia_std = cv_results[metric_key.replace('mean', 'stdv')][best_idx]

    # =========================
    # 4️⃣ Guardar resultados del trial
    # =========================
    os.makedirs("optuna_results", exist_ok=True)
    registro = {
        "trial": trial.number,
        "ganancia_promedio": float(ganancia_promedio),
        "ganancia_std": float(ganancia_std),
        "params": params
    }
    with open(f"optuna_results/trial_{trial.number}.json", "w") as f:
        json.dump(registro, f, indent=4)

    logger.info(f"✅ Trial {trial.number} - Ganancia CV: {ganancia_promedio:,.0f} ± {ganancia_std:,.0f}")

    return ganancia_promedio


# =====================================
# 🚀 Proceso de optimización
# =====================================
def optimizar_con_cv(df, n_trials=50) -> optuna.Study:
    """
    Ejecuta optimización bayesiana con CV tradicional (LightGBM + Optuna).
    """
    logger.info(f"🚀 Iniciando optimización con {n_trials} trials (CV tradicional)")

    study = optuna.create_study(
        direction='maximize',
        study_name=STUDY_NAME,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0]),
        load_if_exists=True
    )

    # Ejecutar la optimización bayesiana
    study.optimize(lambda trial: objetivo_ganancia_cv(trial, df), n_trials=n_trials)

    # =========================
    # 5️⃣ Entrenamiento final con mejores hiperparámetros
    # =========================
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'deterministic': True,
        'bagging_seed': SEMILLA[0],
        'feature_fraction_seed': SEMILLA[0],
        'drop_seed': SEMILLA[0],
        'extra_seed': SEMILLA[0],
        'data_random_seed': SEMILLA[0],
        'num_threads': 1
    })

    df_train = df[df["foto_mes"].isin(MES_TRAIN)]
    X_train = df_train[[c for c in df_train.columns if c != "clase_ternaria"]]
    y_train = df_train["clase_ternaria"]

    dtrain = lgb.Dataset(X_train, label=y_train)

    cv_results_final = lgb.cv(
        best_params,
        dtrain,
        num_boost_round=150,
        nfold=5,
        stratified=True,
        feval=ganancia_evaluator,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        seed=SEMILLA[0]
    )

    best_round = len(cv_results_final[list(cv_results_final.keys())[0]])
    # Sugerido por GPT: Guardar el best_round
    with open(f"resultados/{STUDY_NAME}_best_round.json", "w") as f:
        json.dump({"best_round": best_round}, f)
    model_final = lgb.train(best_params, dtrain, num_boost_round=best_round)

    generar_grafico_importancia(model_final, titulo_estudio=f"Estudio {STUDY_NAME}")

    # =========================
    # 6️⃣ Guardar resultados finales
    # =========================
    resultados = [
        {"trial_number": t.number, "value": t.value, "params": t.params}
        for t in study.trials
    ]

    os.makedirs("resultados", exist_ok=True)
    archivo_final = f"resultados/{STUDY_NAME}_iteraciones.json"
    with open(archivo_final, "w") as f:
        json.dump(resultados, f, indent=4)

    logger.info(f"🏁 Optimización finalizada. Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"🧠 Mejores parámetros: {study.best_params}")

    try:
        generar_graficos_optuna(study, STUDY_NAME)
    except Exception as e:
        logger.warning(f"No se pudieron generar los gráficos de Optuna: {e}")

    return study
