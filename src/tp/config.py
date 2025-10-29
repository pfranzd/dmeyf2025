# src/config.py
import yaml
import os
import logging

logger = logging.getLogger(__name__)

#Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(__file__), "config.yaml")

try:
    with open(PATH_CONFIG, "r") as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral["competencia01"]



        STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "Wendsday")
        DATA_PATH = _cfg.get("DATA_PATH", "../data/competencia.csv")
        SEMILLA = _cfg.get("SEMILLA", [42])
        MES_TRAIN = _cfg.get("MES_TRAIN", "202102")
        MES_VALIDACION = _cfg.get("MES_VALIDACION", "202103")
        MES_TEST = _cfg.get("MES_TEST", "202104")
        GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
        COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", None)
        FINAL_TRAIN = _cfg.get("FINAL_TRAIN", ["202101", "202102", "202103", "202104"])
        FINAL_PREDIC = _cfg.get("FINAL_PREDIC", "202106")
        ATRIBUTOS_FE = _cfg.get("ATRIBUTOS_FE", ["mcuentas_saldo", "mtarjeta_visa_consumo", "cproductos"])

        _cfg_lgb = _cfgGeneral.get("parametros_lgb", {})

        PARAMETROS_LGB = {
            "n_trail": _cfg_lgb.get("n_trail", 50),
            "num_leaves": _cfg_lgb.get("num_leaves", [10, 100]),
            "learning_rate": _cfg_lgb.get("learning_rate", [0.01, 0.1]),
            "feature_fraction": _cfg_lgb.get("feature_fraction", [0.6, 1.0]),
            "bagging_fraction": _cfg_lgb.get("bagging_fraction", [0.6, 1.0]),
            "min_child_samples": _cfg_lgb.get("min_child_samples", [10, 100]),
            "max_depth": _cfg_lgb.get("max_depth", [3, 10]),
            "reg_alpha": _cfg_lgb.get("reg_alpha", [0.0, 5.0]),
            "reg_lambda": _cfg_lgb.get("reg_lambda", [0.0, 5.0]),
            "max_bin": _cfg_lgb.get("max_bin", [30, 31]),
        }

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise
