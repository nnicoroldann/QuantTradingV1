# model_trainer.py - Entrenamiento y evaluación de modelos ML

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import MLConfig
from data_processor import DataProcessor

class ModelTrainer:
    def __init__(self):
        self.config = MLConfig()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Entrena Random Forest"""
        print("Entrenando Random Forest...")
        
        params = self.config.MODELS_TO_TRAIN['random_forest']
        model = RandomForestClassifier(**params)
        
        model.fit(X_train, y_train)
        
        # Predicciones
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Probabilidades
        train_proba = model.predict_proba(X_train)[:, 1]
        val_proba = model.predict_proba(X_val)[:, 1]
        
        return model, {
            'train_pred': train_pred,
            'val_pred': val_pred,
            'train_proba': train_proba,
            'val_proba': val_proba
        }
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Entrena XGBoost"""
        print("Entrenando XGBoost...")
        
        params = self.config.MODELS_TO_TRAIN['xgboost']
        model = xgb.XGBClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Predicciones
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Probabilidades
        train_proba = model.predict_proba(X_train)[:, 1]
        val_proba = model.predict_proba(X_val)[:, 1]
        
        return model, {
            'train_pred': train_pred,
            'val_pred': val_pred,
            'train_proba': train_proba,
            'val_proba': val_proba
        }
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Entrena LightGBM"""
        print("Entrenando LightGBM...")
        
        params = self.config.MODELS_TO_TRAIN['lightgbm']
        model = lgb.LGBMClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        # Predicciones
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Probabilidades
        train_proba = model.predict_proba(X_train)[:, 1]
        val_proba = model.predict_proba(X_val)[:, 1]
        
        return model, {
            'train_pred': train_pred,
            'val_pred': val_pred,
            'train_proba': train_proba,
            'val_proba': val_proba
        }
    
    def calculate_trading_metrics(self, y_true, y_pred, y_proba):
        """Calcula métricas específicas para trading"""
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Simular trading básico
        # Comprar cuando predice subida (1), vender cuando predice bajada (0)
        returns = []
        for i in range(len(y_pred)):
            if y_pred[i] == 1:  # Predicción de subida
                actual_return = 1 if y_true[i] == 1 else -1  # +1 si acertó, -1 si falló
            else:  # Predicción de bajada
                actual_return = 1 if y_true[i] == 0 else -1  # +1 si acertó, -1 si falló
            returns.append(actual_return)
        
        returns = np.array(returns)
        total_trades = len(returns)
        winning_trades = np.sum(returns > 0)
        losing_trades = np.sum(returns < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = winning_trades / losing_trades if losing_trades > 0 else np.inf
        
        # Sharpe ratio simplificado
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = avg_return / std_return if std_return > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }
    
    def evaluate_model(self, model_name, model, predictions, y_train, y_val):
        """Evalúa un modelo entrenado"""
        print(f"Evaluando {model_name}...")
        
        # Extraer predicciones
        train_pred = predictions['train_pred']
        val_pred = predictions['val_pred']
        train_proba = predictions['train_proba']
        val_proba = predictions['val_proba']
        
        # Calcular métricas para entrenamiento
        train_metrics = self.calculate_trading_metrics(y_train, train_pred, train_proba)
        
        # Calcular métricas para validación
        val_metrics = self.calculate_trading_metrics(y_val, val_pred, val_proba)
        
        # Feature importance (si está disponible)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        
        results = {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': feature_importance,
            'predictions': predictions
        }
        
        # Mostrar resultados
        print(f"\n=== RESULTADOS {model_name.upper()} ===")
        print("ENTRENAMIENTO:")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")
        print(f"  F1-Score: {train_metrics['f1']:.4f}")
        print(f"  Win Rate: {train_metrics['win_rate']:.4f}")
        print(f"  Profit Factor: {train_metrics['profit_factor']:.4f}")
        print(f"  Sharpe Ratio: {train_metrics['sharpe_ratio']:.4f}")
        
        print("VALIDACIÓN:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1-Score: {val_metrics['f1']:.4f}")
        print(f"  Win Rate: {val_metrics['win_rate']:.4f}")
        print(f"  Profit Factor: {val_metrics['profit_factor']:.4f}")
        print(f"  Sharpe Ratio: {val_metrics['sharpe_ratio']:.4f}")
        
        return results
    
    def train_all_models(self, data):
        """Entrena todos los modelos configurados"""
        print("=== ENTRENANDO TODOS LOS MODELOS ===")
        
        X_train = data['X_train']
        X_val = data['X_val']
        y_train = data['y_train']
        y_val = data['y_val']
        
        # Random Forest
        if 'random_forest' in self.config.MODELS_TO_TRAIN:
            model, predictions = self.train_random_forest(X_train, y_train, X_val, y_val)
            self.models['random_forest'] = model
            self.results['random_forest'] = self.evaluate_model('random_forest', model, predictions, y_train, y_val)
        
        # XGBoost
        if 'xgboost' in self.config.MODELS_TO_TRAIN:
            model, predictions = self.train_xgboost(X_train, y_train, X_val, y_val)
            self.models['xgboost'] = model
            self.results['xgboost'] = self.evaluate_model('xgboost', model, predictions, y_train, y_val)
        
        # LightGBM
        if 'lightgbm' in self.config.MODELS_TO_TRAIN:
            model, predictions = self.train_lightgbm(X_train, y_train, X_val, y_val)
            self.models['lightgbm'] = model
            self.results['lightgbm'] = self.evaluate_model('lightgbm', model, predictions, y_train, y_val)
    
    def select_best_model(self):
        """Selecciona el mejor modelo basado en métricas de validación"""
        print("\n=== SELECCIONANDO MEJOR MODELO ===")
        
        best_score = -1
        best_model_name = None
        
        print("Comparación de modelos (Validación):")
        print("-" * 80)
        print(f"{'Modelo':<15} {'Accuracy':<10} {'Win Rate':<10} {'Profit F.':<10} {'Sharpe':<10}")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            val_metrics = results['val_metrics']
            
            # Score combinado (puedes ajustar los pesos)
            score = (
                val_metrics['accuracy'] * 0.3 +
                val_metrics['win_rate'] * 0.3 +
                min(val_metrics['profit_factor'], 3.0) / 3.0 * 0.2 +  # Cap profit factor
                val_metrics['sharpe_ratio'] * 0.2
            )
            
            print(f"{model_name:<15} {val_metrics['accuracy']:<10.4f} {val_metrics['win_rate']:<10.4f} {val_metrics['profit_factor']:<10.4f} {val_metrics['sharpe_ratio']:<10.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        print("-" * 80)
        print(f"MEJOR MODELO: {best_model_name} (Score: {best_score:.4f})")
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        return best_model_name, self.best_model
    
    def test_best_model(self, data):
        """Evalúa el mejor modelo en el conjunto de test"""
        if self.best_model is None:
            raise ValueError("Primero debe seleccionar el mejor modelo")
        
        print(f"\n=== EVALUACIÓN FINAL DEL MEJOR MODELO ({self.best_model_name}) ===")
        
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Predicciones en test
        test_pred = self.best_model.predict(X_test)
        test_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Métricas finales
        test_metrics = self.calculate_trading_metrics(y_test, test_pred, test_proba)
        
        print("RESULTADOS EN TEST:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        print(f"  Win Rate: {test_metrics['win_rate']:.4f}")
        print(f"  Profit Factor: {test_metrics['profit_factor']:.4f}")
        print(f"  Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
        print(f"  Total Trades: {test_metrics['total_trades']}")
        print(f"  Winning Trades: {test_metrics['winning_trades']}")
        print(f"  Losing Trades: {test_metrics['losing_trades']}")
        
        # Matriz de confusión
        print(f"\nMatriz de Confusión:")
        cm = confusion_matrix(y_test, test_pred)
        print(cm)
        
        return test_metrics
    
    def save_best_model(self):
        """Guarda el mejor modelo y sus metadatos"""
        if self.best_model is None:
            raise ValueError("No hay modelo para guardar")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar modelo
        model_path = os.path.join(self.config.MODELS_DIR, f'best_model_{self.best_model_name}_{timestamp}.joblib')
        joblib.dump(self.best_model, model_path)
        
        # Guardar modelo como "latest" para uso en producción
        latest_path = os.path.join(self.config.MODELS_DIR, 'latest_model.joblib')
        joblib.dump(self.best_model, latest_path)
        
        # Guardar metadatos
        metadata = {
            'model_name': self.best_model_name,
            'timestamp': timestamp,
            'validation_metrics': self.results[self.best_model_name]['val_metrics'],
            'model_path': model_path
        }
        
        metadata_path = os.path.join(self.config.MODELS_DIR, 'model_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        print(f"\nModelo guardado:")
        print(f"  Archivo: {model_path}")
        print(f"  Modelo actual: {latest_path}")
        print(f"  Metadatos: {metadata_path}")
        
        return model_path
    
    def create_feature_importance_report(self, feature_columns):
        """Crea reporte de importancia de features"""
        if self.best_model is None or not hasattr(self.best_model, 'feature_importances_'):
            print("No hay importancia de features disponible")
            return None
        
        importance = self.best_model.feature_importances_
        
        # Crear DataFrame
        feature_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n=== TOP 20 FEATURES MÁS IMPORTANTES ({self.best_model_name}) ===")
        print(feature_df.head(20).to_string(index=False))
        
        # Guardar reporte completo
        report_path = os.path.join(self.config.RESULTS_DIR, 'feature_importance.csv')
        feature_df.to_csv(report_path, index=False)
        print(f"\nReporte completo guardado en: {report_path}")
        
        return feature_df
    
    def save_training_report(self, test_metrics=None):
        """Guarda reporte completo del entrenamiento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'timestamp': timestamp,
            'config': {
                'lookback_window': self.config.LOOKBACK_WINDOW,
                'prediction_horizon': self.config.PREDICTION_HORIZON,
                'min_price_change': self.config.MIN_PRICE_CHANGE,
                'train_size': self.config.TRAIN_SIZE,
                'val_size': self.config.VAL_SIZE,
                'test_size': self.config.TEST_SIZE
            },
            'models_results': {}
        }
        
        # Agregar resultados de todos los modelos
        for model_name, results in self.results.items():
            report['models_results'][model_name] = {
                'train_metrics': results['train_metrics'],
                'val_metrics': results['val_metrics']
            }
        
        # Agregar mejor modelo y test
        if self.best_model_name:
            report['best_model'] = self.best_model_name
            if test_metrics:
                report['test_metrics'] = test_metrics
        
        # Guardar reporte
        report_path = os.path.join(self.config.RESULTS_DIR, f'training_report_{timestamp}.joblib')
        joblib.dump(report, report_path)
        
        print(f"Reporte de entrenamiento guardado en: {report_path}")
        
        return report_path

def main():
    """Función principal para entrenar modelos"""
    print("=== INICIANDO ENTRENAMIENTO DE MODELOS ML ===")
    
    # Crear instancias
    trainer = ModelTrainer()
    processor = DataProcessor()
    
    # Procesar datos
    print("Procesando datos...")
    data = processor.process_all_data()
    
    # Entrenar modelos
    trainer.train_all_models(data)
    
    # Seleccionar mejor modelo
    best_model_name, best_model = trainer.select_best_model()
    
    # Evaluar en test
    test_metrics = trainer.test_best_model(data)
    
    # Guardar modelo
    model_path = trainer.save_best_model()
    
    # Crear reportes
    trainer.create_feature_importance_report(data['feature_columns'])
    trainer.save_training_report(test_metrics)
    
    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Mejor modelo: {best_model_name}")
    print(f"Archivo del modelo: {model_path}")

if __name__ == "__main__":
    main()