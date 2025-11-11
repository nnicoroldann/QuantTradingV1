# config.py - Configuraciones del sistema ML

import os

class MLConfig:
    # Rutas de archivos
    DATA_DIR = "../output/indicadores/"
    MODELS_DIR = "models/"
    RESULTS_DIR = "results/"
    
    # Configuración de datos
    ASSETS = ["indicadores_GGAL.BA", "indicadores_VIST.BA", "indicadores_YPFD.BA"]  # Cambia por tus activos reales
    CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "num_trades", "tipo", 
               "rsi", "bb_middle", "bb_upper", "bb_lower", "bb_width", "adx", 
               "di_plus", "di_minus", "volume_sma", "volume_ratio"]
    
    # Parámetros de features
    LOOKBACK_WINDOW = 30  # Cuántas velas hacia atrás usar para predicción
    
    # Indicadores técnicos - períodos
    SMA_PERIODS = [5, 10, 20, 50]
    EMA_PERIODS = [5, 10, 20]
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    
    # Parámetros de predicción
    PREDICTION_HORIZON = 1  # Predecir 1 vela hacia adelante
    MIN_PRICE_CHANGE = 0.005  # 0.5% mínimo cambio para considerar subida/bajada
    
    # División de datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    
    # Parámetros de modelos
    MODELS_TO_TRAIN = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 10,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }
    
    # Crear directorios si no existen
    @staticmethod
    def create_directories():
        for directory in [MLConfig.MODELS_DIR, MLConfig.RESULTS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)