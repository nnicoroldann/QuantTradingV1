# predictor.py - Predictor en tiempo real para trading

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import MLConfig
from feature_engineering import FeatureEngineer

class TradingPredictor:
    def __init__(self, model_path=None):
        self.config = MLConfig()
        self.feature_engineer = FeatureEngineer()
        
        # Cargar modelo y componentes
        self.model = None
        self.scaler = None
        self.feature_info = None
        
        self.load_model_components(model_path)
    
    def load_model_components(self, model_path=None):
        """Carga el modelo y componentes necesarios"""
        try:
            # Usar modelo especificado o el último guardado
            if model_path is None:
                model_path = os.path.join(self.config.MODELS_DIR, 'latest_model.joblib')
            
            # Cargar modelo
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"Modelo cargado desde: {model_path}")
            else:
                raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
            
            # Cargar scaler
            scaler_path = os.path.join(self.config.MODELS_DIR, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("Scaler cargado correctamente")
            else:
                raise FileNotFoundError(f"No se encontró el scaler en: {scaler_path}")
            
            # Cargar información de features
            feature_info_path = os.path.join(self.config.MODELS_DIR, 'feature_info.joblib')
            if os.path.exists(feature_info_path):
                self.feature_info = joblib.load(feature_info_path)
                print("Información de features cargada correctamente")
            else:
                raise FileNotFoundError(f"No se encontró feature_info en: {feature_info_path}")
            
            # Cargar metadatos del modelo
            metadata_path = os.path.join(self.config.MODELS_DIR, 'model_metadata.joblib')
            if os.path.exists(metadata_path):
                self.metadata = joblib.load(metadata_path)
                print(f"Metadatos cargados - Modelo: {self.metadata['model_name']}")
            
        except Exception as e:
            print(f"Error cargando componentes del modelo: {e}")
            raise
    
    def load_recent_data(self, csv_path, n_candles=None):
        """Carga las últimas velas del CSV"""
        if n_candles is None:
            n_candles = self.config.LOOKBACK_WINDOW + 10  # Margen extra
        
        try:
            # Leer CSV
            df = pd.read_csv(csv_path)
            
            # Ordenar por timestamp y tomar las últimas
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').tail(n_candles).reset_index(drop=True)
            
            print(f"Cargadas {len(df)} velas más recientes desde {csv_path}")
            
            return df
            
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return None
    
    def prepare_prediction_data(self, df):
        """Prepara los datos para predicción"""
        try:
            # Agregar columna de activo si no existe
            if 'asset' not in df.columns:
                df['asset'] = 'unknown'
            
            # Crear features
            df_features = self.feature_engineer.create_all_features(df)
            
            # Verificar que tenemos suficientes datos
            if len(df_features) < self.config.LOOKBACK_WINDOW:
                raise ValueError(f"Necesitas al menos {self.config.LOOKBACK_WINDOW} velas con features válidas")
            
            # Tomar la última secuencia
            feature_columns = self.feature_info['feature_columns']
            
            # Verificar que todas las columnas existen
            missing_cols = set(feature_columns) - set(df_features.columns)
            if missing_cols:
                raise ValueError(f"Faltan columnas de features: {missing_cols}")
            
            # Crear secuencia
            last_sequence = df_features[feature_columns].tail(self.config.LOOKBACK_WINDOW).values
            
            # Aplanar para modelos tradicionales
            X = last_sequence.flatten().reshape(1, -1)
            
            # Escalar
            X_scaled = self.scaler.transform(X)
            
            return X_scaled, df_features.tail(1)  # Retornar también la última fila para contexto
            
        except Exception as e:
            print(f"Error preparando datos: {e}")
            return None, None
    
    def predict_next_movement(self, csv_path):
        """Predice el próximo movimiento del precio"""
        try:
            # Cargar datos recientes
            df = self.load_recent_data(csv_path)
            if df is None:
                return None
            
            # Preparar datos
            X_scaled, last_candle = self.prepare_prediction_data(df)
            if X_scaled is None:
                return None
            
            # Hacer predicción
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            # Interpretar resultados
            direction = "SUBIDA" if prediction == 1 else "BAJADA"
            confidence = max(probability)
            
            # Información adicional
            current_price = last_candle['close'].iloc[0]
            timestamp = last_candle['timestamp'].iloc[0]
            
            result = {
                'timestamp': timestamp,
                'current_price': current_price,
                'prediction': int(prediction),
                'direction': direction,
                'confidence': confidence,
                'probability_up': probability[1],
                'probability_down': probability[0],
                'model_used': self.metadata['model_name'] if hasattr(self, 'metadata') else 'unknown'
            }
            
            return result
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            return None
    
    def predict_multiple_assets(self, csv_paths):
        """Predice para múltiples activos"""
        results = {}
        
        for asset_name, csv_path in csv_paths.items():
            print(f"\nPrediciendo para {asset_name}...")
            prediction = self.predict_next_movement(csv_path)
            results[asset_name] = prediction
        
        return results
    
    def print_prediction(self, prediction_result, asset_name=""):
        """Imprime la predicción de forma legible"""
        if prediction_result is None:
            print(f"No se pudo obtener predicción para {asset_name}")
            return
        
        print(f"\n=== PREDICCIÓN {asset_name.upper()} ===")
        print(f"Timestamp: {prediction_result['timestamp']}")
        print(f"Precio Actual: ${prediction_result['current_price']:.4f}")
        print(f"Predicción: {prediction_result['direction']}")
        print(f"Confianza: {prediction_result['confidence']:.2%}")
        print(f"Probabilidad Subida: {prediction_result['probability_up']:.2%}")
        print(f"Probabilidad Bajada: {prediction_result['probability_down']:.2%}")
        print(f"Modelo: {prediction_result['model_used']}")
        
        # Recomendación simple
        if prediction_result['confidence'] > 0.6:
            if prediction_result['direction'] == "SUBIDA":
                print("🟢 RECOMENDACIÓN: Considerar posición LARGA")
            else:
                print("🔴 RECOMENDACIÓN: Considerar posición CORTA")
        else:
            print("🟡 RECOMENDACIÓN: Señal débil, esperar o no operar")
    
    def save_prediction_log(self, predictions, log_file="prediction_log.csv"):
        """Guarda las predicciones en un log"""
        log_path = os.path.join(self.config.RESULTS_DIR, log_file)
        
        # Preparar datos para el log
        log_data = []
        current_time = datetime.now()
        
        for asset, pred in predictions.items():
            if pred is not None:
                log_data.append({
                    'prediction_timestamp': current_time,
                    'asset': asset,
                    'data_timestamp': pred['timestamp'],
                    'current_price': pred['current_price'],
                    'prediction': pred['prediction'],
                    'direction': pred['direction'],
                    'confidence': pred['confidence'],
                    'probability_up': pred['probability_up'],
                    'probability_down': pred['probability_down'],
                    'model_used': pred['model_used']
                })
        
        if log_data:
            df_log = pd.DataFrame(log_data)
            
            # Agregar al log existente o crear nuevo
            if os.path.exists(log_path):
                existing_log = pd.read_csv(log_path)
                df_log = pd.concat([existing_log, df_log], ignore_index=True)
            
            df_log.to_csv(log_path, index=False)
            print(f"Predicciones guardadas en: {log_path}")
    
    def get_model_info(self):
        """Retorna información del modelo cargado"""
        info = {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'feature_info_loaded': self.feature_info is not None
        }
        
        if hasattr(self, 'metadata'):
            info.update(self.metadata)
        
        if self.feature_info:
            info['n_features'] = self.feature_info['n_features']
            info['lookback_window'] = self.feature_info['lookback_window']
        
        return info

def main():
    """Función principal para hacer predicciones"""
    print("=== PREDICTOR DE TRADING ML ===")
    
    # Crear predictor
    predictor = TradingPredictor()
    
    # Mostrar info del modelo
    model_info = predictor.get_model_info()
    print(f"Modelo cargado: {model_info.get('model_name', 'unknown')}")
    
    # Ejemplo de uso - actualizar con tus paths reales
    csv_paths = {
        'BTCUSDT': 'data/btc.csv',
        'ETHUSDT': 'data/eth.csv', 
        'ADAUSDT': 'data/ada.csv'
    }
    
    try:
        # Hacer predicciones para todos los activos
        predictions = predictor.predict_multiple_assets(csv_paths)
        
        # Mostrar resultados
        for asset, prediction in predictions.items():
            predictor.print_prediction(prediction, asset)
        
        # Guardar log
        predictor.save_prediction_log(predictions)
        
    except Exception as e:
        print(f"Error en predicciones: {e}")

if __name__ == "__main__":
    main()