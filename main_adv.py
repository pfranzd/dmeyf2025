import pandas as pd
import os
import random
import datetime
import numpy as np
import logging
import json

from src.tp.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.tp.features import feature_engineering_lag, feature_engineering_business, feature_engineering_moving_avg, feature_engineering_business_alternative
from src.tp.optimization import optimizar, evaluar_en_test, guardar_resultados_test
from src.tp.optimizacion_cv import optimizar_con_cv
from src.tp.best_params import cargar_mejores_hiperparametros
from src.tp.final_training import *
from src.tp.output_manager import guardar_predicciones_finales
from src.tp.config import *
from src.tp.grafico_test import crear_grafico_ganancia_avanzado

## setear semillas
random.seed(SEMILLA[0])
np.random.seed(SEMILLA[0])
os.environ["PYTHONHASHSEED"] = str(SEMILLA[0])

## config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

## Parámetros globales
optimization = False

## Funcion principal
def main():
    logger.info("Inicio de ejecucion.")

    #00 Cargar datos
    os.makedirs("data", exist_ok=True)
    df = cargar_datos(DATA_PATH)

    # 2. Feature Engineering
    # 2.1 Business Feature Engineering

    df_fe = feature_engineering_business_alternative(df).copy()

    # 2.2 Time Feature Engineering
    
    atributos = ATRIBUTOS_FE
    cant_lag = 2
    df_fe = feature_engineering_lag(df_fe, atributos, cant_lag)
    df_fe = feature_engineering_moving_avg(df_fe, atributos, cant_lag)
    logger.info(f"Feature Engineering completado: {df_fe.shape}")

    # Crear copia para test
    df_fe_test = df_fe.copy()

    #02 Convertir clase_ternaria a target binario
    df_fe = convertir_clase_ternaria_a_target(df_fe, True) # True = BAJA+1 y BAJA+2
    df_fe_test = convertir_clase_ternaria_a_target(df_fe_test, False) # False = solo BAJA+2
  
    if optimization:
        #03 Ejecutar optimizacion de hiperparametros
        study = optimizar_con_cv(df_fe, n_trials=30)
    
        #04 Análisis adicional
        logger.info("=== ANÁLISIS DE RESULTADOS ===")
        trials_df = study.trials_dataframe()
        if len(trials_df) > 0:
            top_5 = trials_df.nlargest(5, 'value')
            logger.info("Top 5 mejores trials:")
            for idx, trial in top_5.iterrows():
                logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
    
        logger.info("=== OPTIMIZACIÓN COMPLETADA ===")
  
    #05 Test en mes desconocido
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    # Cargar mejores hiperparámetros
    mejores_params = cargar_mejores_hiperparametros()
  
    # Evaluar en test
    # resultados_test, y_pred_proba = evaluar_en_test(df_fe_test, mejores_params)
    resultados_test, y_pred_proba, y_test = evaluar_en_test(df_fe_test, mejores_params)
  
    # Guardar resultados de test
    guardar_resultados_test(resultados_test)
  
    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"✅ Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    logger.info(f"🎯 Predicciones positivas: {resultados_test['predicciones_positivas']:,} ({resultados_test['porcentaje_positivas']:.2f}%)")

    # Grafico de test
    # logger.info("=== GRAFICO DE TEST ===")
    # ruta_grafico = crear_grafico_ganancia_avanzado(np.array(df_fe['clase_ternaria']), np.array(y_pred_proba))
    ruta_grafico = crear_grafico_ganancia_avanzado(y_test, y_pred_proba)
    logger.info(f"✅ Gráfico generado: {ruta_grafico}")
  
    modelo_path = f"modelos/{STUDY_NAME}_final_model.txt"
    round_path = f"modelos/{STUDY_NAME}_best_round.json"
    params_path = f"modelos/{STUDY_NAME}_best_params.json"

    if os.path.exists(modelo_path) and os.path.exists(round_path) and os.path.exists(params_path):
        logger.info(f"📦 Modelo entrenado previamente detectado: {modelo_path}")
        modelo_final = lgb.Booster(model_file=modelo_path)
        with open(params_path, "r") as f:
            mejores_params = json.load(f)
        with open(round_path, "r") as f:
            best_round = json.load(f)["best_round"]
        logger.info(f"✅ Modelo cargado (best_round={best_round})")
        usar_modelo_existente = True
    else:
        usar_modelo_existente = False

    #06 Entrenar modelo final
    logger.info("=== ENTRENAMIENTO FINAL ===")
    logger.info("Preparar datos para entrenamiento final")
    X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_fe)
  
    # Entrenar modelo final
    logger.info("Entrenar modelo final")
    # Caso con una sola semilla
    # modelo_final = entrenar_modelo_final(X_train, y_train, mejores_params)
    
    # Caso con múltiples semillas y ensamble
    if usar_modelo_existente:
        # Ya cargamos modelo_final arriba
        modelos_ensemble = [modelo_final]
    else:
        # Entrenar nuevo modelo y guardarlo
        logger.info("Entrenar modelo final (nuevo entrenamiento)")
        modelos_ensemble = entrenar_modelos_multiples_semillas_grafico_qa(X_train, y_train, mejores_params)

        # Si usas ensemble de varios seeds, puedes guardar solo el modelo promedio principal
        os.makedirs("modelos", exist_ok=True)
        modelos_ensemble[0].save_model(modelo_path)
        with open(params_path, "w") as f:
            json.dump(mejores_params, f, indent=4)
        with open(round_path, "w") as f:
            json.dump({"best_round": modelos_ensemble[0].best_iteration}, f)

        logger.info(f"📦 Modelo final guardado en {modelo_path}")
  
    # Generar predicciones finales
    logger.info("Generar predicciones finales")

    # Caso con una sola semilla
    # resultados = generar_predicciones_finales(modelo_final, X_predict, clientes_predict, umbral=10000, tipo_umbral='cantidad')
    # Caso con múltiples semillas y ensamble
    resultados = generar_predicciones_ensamble(modelos_ensemble, X_predict, clientes_predict, umbral=11000, tipo_umbral='cantidad')
  
    # Guardar predicciones
    logger.info("Guardar predicciones")
    archivo_salida = guardar_predicciones_finales(resultados)
  
    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info(f"✅ Entrenamiento final completado exitosamente")
    logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
    logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"🔮 Período de predicción: {FINAL_PREDIC}")
    logger.info(f"📁 Archivo de salida: {archivo_salida}")
    logger.info(f"📝 Log detallado: logs/{nombre_log}")


    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")

if __name__ == "__main__":
    main()