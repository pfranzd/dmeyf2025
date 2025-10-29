# src/main_final.py
import os
import datetime
import logging
import numpy as np
import pandas as pd

from src.tp.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.tp.features import (
    feature_engineering_business_alternative,
    feature_engineering_lag,
    feature_engineering_moving_avg
)
from src.tp.best_params import cargar_mejores_hiperparametros
from src.tp.final_training import (
    preparar_datos_entrenamiento_final,
    entrenar_modelos_multiples_semillas,
    generar_predicciones_ensamble
)
from src.tp.output_manager import guardar_predicciones_finales
from src.tp.config import *

# ==============================
# CONFIGURACIÓN DE LOGGING
# ==============================
os.makedirs("logs", exist_ok=True)
fecha = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
nombre_log = f"log_FINAL_{STUDY_NAME}_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# FUNCIÓN PRINCIPAL
# ==============================
def main():
    logger.info("🚀 Inicio de ejecución final para generación de submission Kaggle")

    # 1️⃣ CARGAR DATOS
    df = cargar_datos(DATA_PATH)
    logger.info(f"Datos cargados: {df.shape}")

    # 2️⃣ FEATURE ENGINEERING
    logger.info("Aplicando Feature Engineering de negocio...")
    df_fe = feature_engineering_business_alternative(df)

    logger.info("Aplicando Feature Engineering temporal (lags y medias móviles)...")
    df_fe = feature_engineering_lag(df_fe, ATRIBUTOS_FE, cant_lag=2)
    df_fe = feature_engineering_moving_avg(df_fe, ATRIBUTOS_FE, cant_ventanas=2)
    logger.info(f"Feature Engineering completado: {df_fe.shape}")

    # 3️⃣ CONVERTIR TARGET
    logger.info("Convirtiendo clase_ternaria a target binario (BAJA+1 y BAJA+2 como positivo)")
    df_fe = convertir_clase_ternaria_a_target(df_fe, True)

    # 4️⃣ CARGAR MEJORES HIPERPARÁMETROS
    mejores_params = {'n_trail': 100,
                      'num_leaves': 154, 
                      'learning_rate': 0.08080774514176184, 
                      'feature_fraction': 0.539006047268895, 
                      'bagging_fraction': 0.43632415959978765, 
                      'min_child_samples': 93, 
                      'max_depth': 18, 
                      'reg_alpha': 1.660210153306255, 
                      'reg_lambda': 2.028983357996869, 
                      'max_bin': 31}
    
    mejores_params.update({"objective": "binary", "metric": "custom"})
    logger.info(f"Hiperparámetros óptimos cargados: {mejores_params}")

    # 5️⃣ PREPARAR DATOS DE ENTRENAMIENTO FINAL
    X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_fe)
    logger.info(f"Datos preparados para entrenamiento: X_train={X_train.shape}, X_pred={X_predict.shape}")

    # 6️⃣ ENTRENAMIENTO FINAL (Ensemble multi-semilla)
    logger.info("Entrenando modelos finales (ensamble con múltiples semillas)...")
    modelos_ensemble = entrenar_modelos_multiples_semillas(X_train, y_train, mejores_params)
    logger.info(f"Entrenamiento completado: {len(modelos_ensemble)} modelos generados")

    # 7️⃣ GENERAR PREDICCIONES FINALES
    logger.info("Generando predicciones finales para Kaggle...")
    resultados = generar_predicciones_ensamble(
        modelos_ensemble,
        X_predict,
        clientes_predict,
        umbral=10000,              # Top X clientes (por cantidad)
        tipo_umbral='cantidad'     # 'probabilidad' o 'cantidad'
    )

    # 8️⃣ GUARDAR SUBMISSION
    archivo_salida = guardar_predicciones_finales(resultados)
    logger.info(f"✅ Archivo de submission generado: {archivo_salida}")

    logger.info("🎯 Proceso finalizado correctamente. Listo para subir a Kaggle.")
    logger.info(f"📝 Log: logs/{nombre_log}")

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()