# src/final_training.py
import pandas as pd
import lightgbm as lgb
import matplotlib .pyplot as plt
import numpy as np
import logging
import os
from datetime import datetime
from .config import FINAL_TRAIN, FINAL_PREDIC, SEMILLA, STUDY_NAME
from .best_params import cargar_mejores_hiperparametros
from .grafico_test import calcular_ganancia_acumulada_optimizada
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

    SEMILLA_NUEVA = SEMILLA # + [111111, 222222, 333333, 444444, 555555]

    logger.info(f"Entrenando {len(SEMILLA_NUEVA)} modelos con distintas semillas para ensamblado...")
    for i, seed in enumerate(SEMILLA_NUEVA, start=1):
        logger.info(f"🔁 Entrenando modelo {i}/{len(SEMILLA_NUEVA)} con seed={seed}")

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
            'deterministic': False 
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


def entrenar_modelos_multiples_semillas_grafico_qa(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    mejores_params: dict
) -> list:
    """
    Entrena múltiples modelos de LightGBM usando distintas semillas definidas en SEMILLA.
    Además genera un gráfico comparativo de ganancia acumulada para evaluar estabilidad
    del modelo y facilitar la elección de un umbral adecuado.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento.
        y_train (pd.Series): Target de entrenamiento.
        mejores_params (dict): Mejores hiperparámetros obtenidos con Optuna.
    
    Returns:
        list[lgb.Booster]: Lista de modelos entrenados con distintas semillas.
    """
    modelos = []
    SEMILLA_NUEVA = SEMILLA + [111111, 222222, 333333, 444444, 555555]
    logger.info(f"Entrenando {len(SEMILLA_NUEVA)} modelos con distintas semillas para ensamblado...")

    # Para graficar
    curvas_ganancia = []
    seeds_usadas = []

    for i, seed in enumerate(SEMILLA_NUEVA, start=1):
        logger.info(f"🔁 Entrenando modelo {i}/{len(SEMILLA_NUEVA)} con seed={seed}")

        params = {
            'objective': 'binary',
            'metric': 'None',
            'random_state': seed,
            'verbose': -1,
            **mejores_params,

            # 👇 NUEVO: introduce aleatoriedad
            'bagging_fraction': 0.6,        # usa solo el 80% de las filas
            'bagging_freq': 1,              # aplica bagging en cada iteración
            'feature_fraction': 0.6,        # usa solo el 80% de las columnas
            'bagging_seed': seed,
            'feature_fraction_seed': seed,
            'data_random_seed': seed,
            'drop_seed': seed,
            'deterministic': True,          # permite que la seed afecte el muestreo
            'force_row_wise': True,
            'extra_seed': seed
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        modelo = lgb.train(params=params, train_set=train_data, num_boost_round=40)
        modelos.append(modelo)

        # Predicciones y ganancia acumulada para graficar
        y_pred = modelo.predict(X_train)
        # ganancias, indices_ordenados, _ = calcular_ganancia_acumulada_optimizada(y_train.values, y_pred)
        y_true_array = y_train.values if hasattr(y_train, "values") else y_train
        ganancias, indices_ordenados, _ = calcular_ganancia_acumulada_optimizada(y_true_array, y_pred)
        curvas_ganancia.append(ganancias)
        seeds_usadas.append(seed)

        logger.info(f"✅ Modelo {i} entrenado con éxito (seed={seed})")

    # ========================
    # GRAFICAR CURVAS
    # ========================
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(15, 8))

    # Determinar largo máximo (por si alguna curva difiere levemente)
    max_len = max(len(g) for g in curvas_ganancia)

    for seed, curva in zip(seeds_usadas, curvas_ganancia):
        x = np.arange(len(curva))
        ax.plot(x, curva, label=f'Seed {seed}', linewidth=2, alpha=0.8)

    # Promedio de curvas
    curvas_matrix = np.zeros((len(curvas_ganancia), max_len))
    for i, curva in enumerate(curvas_ganancia):
        curvas_matrix[i, :len(curva)] = curva
    curva_promedio = np.mean(curvas_matrix, axis=0)
    ax.plot(np.arange(max_len), curva_promedio, color='black', linewidth=3, linestyle='--', label='Promedio')

    # Puntos relevantes
    ganancia_max_promedio = np.max(curva_promedio)
    idx_max_promedio = np.argmax(curva_promedio)
    ax.scatter(idx_max_promedio, ganancia_max_promedio, color='red', s=120, zorder=5)
    ax.axvline(idx_max_promedio, color='red', linestyle='--', alpha=0.7)
    ax.annotate(f'Promedio Máx: {ganancia_max_promedio:,.0f}\nCorte: {idx_max_promedio:,}',
                xy=(idx_max_promedio, ganancia_max_promedio),
                xytext=(idx_max_promedio + max_len*0.1, ganancia_max_promedio*1.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='red'))

    ax.set_xlabel('Clientes ordenados por probabilidad', fontsize=12)
    ax.set_ylabel('Ganancia Acumulada', fontsize=12)
    ax.set_title(f'Curvas de Ganancia Acumulada - Ensemble ({len(SEMILLA_NUEVA)} semillas)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Guardar gráfico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("resultados", exist_ok=True)
    ruta_grafico = f"resultados/{STUDY_NAME}_curvas_ensemble_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"📈 Gráfico de curvas de ganancia guardado: {ruta_grafico}")

    return modelos

def entrenar_modelos_multiples_semillas_grafico(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    mejores_params: dict
) -> list:
    """
    Entrena múltiples modelos de LightGBM usando distintas semillas definidas en SEMILLA.
    Además genera un gráfico comparativo de ganancia acumulada para evaluar estabilidad
    del modelo y facilitar la elección de un umbral adecuado.
    
    También inserta en la imagen un resumen estadístico de la ganancia máxima alcanzada
    por cada semilla (mínimo, máximo, media y mediana).

    Returns:
        list[lgb.Booster]: Lista de modelos entrenados con distintas semillas.
    """
    modelos = []
    SEMILLA_NUEVA = SEMILLA + [111111, 222222, 333333, 444444, 555555]
    logger.info(f"Entrenando {len(SEMILLA_NUEVA)} modelos con distintas semillas para ensamblado...")

    curvas_ganancia = []
    seeds_usadas = []
    ganancias_maximas = []
    cortes_maximos = []

    # ============= ENTRENAR MODELOS =============
    for i, seed in enumerate(SEMILLA_NUEVA, start=1):
        logger.info(f"🔁 Entrenando modelo {i}/{len(SEMILLA_NUEVA)} con seed={seed}")

        params = {
            'objective': 'binary',
            'metric': 'None',
            'random_state': seed,
            'verbose': -1,
            **mejores_params,
            # overrides para asegurar aleatoriedad
            'bagging_fraction': 0.6,
            'bagging_freq': 1,
            'feature_fraction': 0.6,
            'bagging_seed': seed,
            'feature_fraction_seed': seed,
            'data_random_seed': seed,
            'drop_seed': seed,
            'extra_seed': seed,
            'deterministic': False
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        modelo = lgb.train(params=params, train_set=train_data, num_boost_round=40)
        modelos.append(modelo)

        # =================== GANANCIAS ===================
        y_pred = modelo.predict(X_train)
        y_true_array = y_train.values if hasattr(y_train, "values") else y_train
        ganancias, indices_ordenados, _ = calcular_ganancia_acumulada_optimizada(y_true_array, y_pred)
        curvas_ganancia.append(ganancias)
        seeds_usadas.append(seed)

        gan_max = np.max(ganancias)
        idx_max = np.argmax(ganancias)
        ganancias_maximas.append(gan_max)
        cortes_maximos.append(idx_max)

        logger.info(f"✅ Modelo {i} (seed={seed}) → Ganancia máx: {gan_max:,.0f} en corte {idx_max:,}")

    # ======================== GRAFICAR CURVAS ========================
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(15, 8))

    max_len = max(len(g) for g in curvas_ganancia)
    for seed, curva in zip(seeds_usadas, curvas_ganancia):
        x = np.arange(len(curva))
        ax.plot(x, curva, label=f'Seed {seed}', linewidth=2, alpha=0.8)

    # Promedio
    curvas_matrix = np.zeros((len(curvas_ganancia), max_len))
    for i, curva in enumerate(curvas_ganancia):
        curvas_matrix[i, :len(curva)] = curva
    curva_promedio = np.mean(curvas_matrix, axis=0)
    ax.plot(np.arange(max_len), curva_promedio, color='black', linewidth=3, linestyle='--', label='Promedio')

    # Punto máximo promedio
    ganancia_max_promedio = np.max(curva_promedio)
    idx_max_promedio = np.argmax(curva_promedio)
    ax.scatter(idx_max_promedio, ganancia_max_promedio, color='red', s=120, zorder=5)
    ax.axvline(idx_max_promedio, color='red', linestyle='--', alpha=0.7)
    ax.annotate(f'Promedio Máx: {ganancia_max_promedio:,.0f}\nCorte: {idx_max_promedio:,}',
                xy=(idx_max_promedio, ganancia_max_promedio),
                xytext=(idx_max_promedio + max_len*0.1, ganancia_max_promedio*1.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='red'))

    ax.set_xlabel('Clientes ordenados por probabilidad', fontsize=12)
    ax.set_ylabel('Ganancia Acumulada', fontsize=12)
    ax.set_title(f'Curvas de Ganancia Acumulada - Ensemble ({len(SEMILLA_NUEVA)} semillas)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # ======================== CUADRO ESTADÍSTICO ========================
    # Cálculo de stats
    gan_min = np.min(ganancias_maximas)
    gan_max = np.max(ganancias_maximas)
    gan_mean = np.mean(ganancias_maximas)
    gan_median = np.median(ganancias_maximas)

    stats_text = (
        f"Resumen Ganancia Máxima por Semilla\n"
        f"-------------------------------------\n"
        f"Min: {gan_min:,.0f}\n"
        f"Max: {gan_max:,.0f}\n"
        f"Media: {gan_mean:,.0f}\n"
        f"Mediana: {gan_median:,.0f}\n\n"
        f"Semillas:\n" + "\n".join([f"{s}: {g:,.0f}" for s, g in zip(seeds_usadas, ganancias_maximas)])
    )

    # Insertar texto en la esquina superior derecha
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(
        0.98, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        va='top', ha='right',
        bbox=props,
        family='monospace'
    )

    # ======================== GUARDAR ========================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("resultados", exist_ok=True)
    ruta_grafico = f"resultados/{STUDY_NAME}_curvas_ensemble_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Log adicional
    logger.info("📊 Estadísticas de ganancia máxima por semilla:")
    for s, g, c in zip(seeds_usadas, ganancias_maximas, cortes_maximos):
        logger.info(f"  Seed {s}: Ganancia máx {g:,.0f} en corte {c:,}")
    logger.info(f"  ➡️ Min: {gan_min:,.0f} | Max: {gan_max:,.0f} | Media: {gan_mean:,.0f} | Mediana: {gan_median:,.0f}")
    logger.info(f"📈 Gráfico de curvas de ganancia guardado en: {ruta_grafico}")

    return modelos