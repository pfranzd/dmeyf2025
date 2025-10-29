# src/best_params.py
import json
import logging
from .config import STUDY_NAME, SEMILLA

logger = logging.getLogger(__name__)

def cargar_mejores_hiperparametros(archivo_base: str = None) -> dict:
    """
    Carga los mejores hiperparámetros desde el archivo JSON de iteraciones de Optuna
    y completa con parámetros deterministas adicionales para LightGBM.
  
    Args:
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)
  
    Returns:
        dict: Mejores hiperparámetros listos para entrenamiento determinista
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)
  
        if not iteraciones:
            raise ValueError("No se encontraron iteraciones en el archivo")
  
        # 1️⃣ Seleccionar la mejor iteración por ganancia
        mejor_iteracion = max(iteraciones, key=lambda x: x['value'])
        mejores_params = mejor_iteracion['params']
        mejor_ganancia = mejor_iteracion['value']
  
        logger.info(f"Mejores hiperparámetros cargados desde {archivo}")
        logger.info(f"Mejor ganancia encontrada: {mejor_ganancia:,.0f}")
        logger.info(f"Trial número: {mejor_iteracion['trial_number']}")
        logger.info(f"Parámetros base Optuna: {mejores_params}")
  
        # 2️⃣ Completar con configuración determinista
        seed = SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA
        mejores_params.update({
            'seed': seed,
            'bagging_seed': seed,
            'feature_fraction_seed': seed,
            'data_random_seed': seed,
            'drop_seed': seed,
            'extra_seed': seed,
            'deterministic': True,
            'force_row_wise': True,
            'num_threads': 1
        })
  
        logger.info(f"Parámetros finales (deterministas): {mejores_params}")
  
        return mejores_params
  
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo {archivo}")
        logger.error("Asegúrate de haber ejecutado la optimización con Optuna primero")
        raise
    except Exception as e:
        logger.error(f"Error al cargar mejores hiperparámetros: {e}")
        raise

def obtener_estadisticas_optuna(archivo_base=None):
    """
    Obtiene estadísticas de la optimización de Optuna.
  
    Args:
        archivo_base: Nombre base del archivo
  
    Returns:
        dict: Estadísticas de la optimización
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)
  
        ganancias = [iter['value'] for iter in iteraciones]
  
        estadisticas = {
            'total_trials': len(iteraciones),
            'mejor_ganancia': max(ganancias),
            'peor_ganancia': min(ganancias),
            'ganancia_promedio': sum(ganancias) / len(ganancias),
            'top_5_trials': sorted(iteraciones, key=lambda x: x['value'], reverse=True)[:5]
        }
  
        logger.info("Estadísticas de optimización:")
        logger.info(f"  Total trials: {estadisticas['total_trials']}")
        logger.info(f"  Mejor ganancia: {estadisticas['mejor_ganancia']:,.0f}")
        logger.info(f"  Ganancia promedio: {estadisticas['ganancia_promedio']:,.0f}")
  
        return estadisticas
  
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        raise