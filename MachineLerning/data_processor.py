# data_processor.py - Procesador de datos para ML

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from config import MLConfig
from feature_engineering import FeatureEngineer

class DataProcessor:
    def __init__(self):
        self.config = MLConfig()
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def load_csv_data(self, csv_path):
        """Carga datos desde CSV"""
        print(f"Cargando datos desde: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Verificar que tiene las columnas esperadas
        expected_cols = self.config.CSV_COLUMNS
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Faltan columnas en el CSV: {missing_cols}")
        
        print(f"Datos cargados: {len(df)} filas")
        return df
    
    def load_all_assets_data(self):
        """Carga datos de todos los activos y los combina"""
        all_data = []
        
        for asset in self.config.ASSETS:
            csv_path = os.path.join(self.config.DATA_DIR, f"{asset}.csv")
            if os.path.exists(csv_path):
                df = self.load_csv_data(csv_path)
                df['asset'] = asset  # Agregar columna de activo
                all_data.append(df)
            else:
                print(f"Advertencia: No se encontró {csv_path}")
        
        if not all_data:
            raise ValueError("No se encontraron archivos de datos")
        
        # Combinar todos los datos
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Datos combinados: {len(combined_df)} filas de {len(all_data)} activos")
        
        return combined_df
    
    def create_target_variable(self, df):
        """Crea la variable objetivo (target) para predicción"""
        print("Creando variable objetivo...")
        
        df = df.copy()
        df = df.sort_values(['asset', 'timestamp']).reset_index(drop=True)
        
        # Calcular precio futuro
        df['future_close'] = df.groupby('asset')['close'].shift(-self.config.PREDICTION_HORIZON)
        
        # Calcular cambio porcentual
        df['price_change'] = (df['future_close'] - df['close']) / df['close']
        
        # Crear target binario
        df['target_binary'] = np.where(
            df['price_change'] > self.config.MIN_PRICE_CHANGE, 1,  # Subida
            np.where(df['price_change'] < -self.config.MIN_PRICE_CHANGE, 0, np.nan)  # Bajada o sin cambio significativo
        )
        
        # Eliminar filas sin target válido
        df = df.dropna(subset=['target_binary'])
        df['target'] = df['target_binary'].astype(int)
        
        # Estadísticas del target
        target_counts = df['target'].value_counts()
        print(f"Distribución del target:")
        print(f"  Bajadas (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  Subidas (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
        
        return df
    
    def create_sequences(self, df):
        """Crea secuencias de datos para el modelo"""
        print("Creando secuencias temporales...")
        
        sequences = []
        targets = []
        
        # Agrupar por activo para mantener secuencias coherentes
        for asset in df['asset'].unique():
            asset_df = df[df['asset'] == asset].sort_values('timestamp').reset_index(drop=True)
            
            # Crear secuencias con ventana deslizante
            for i in range(self.config.LOOKBACK_WINDOW, len(asset_df)):
                # Secuencia de features
                sequence = asset_df.iloc[i-self.config.LOOKBACK_WINDOW:i][self.feature_columns].values
                
                # Target correspondiente
                target = asset_df.iloc[i]['target']
                
                sequences.append(sequence.flatten())  # Aplanar para modelos tradicionales
                targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"Secuencias creadas: {X.shape[0]} muestras, {X.shape[1]} features")
        
        return X, y
    
    def prepare_data_for_training(self, csv_paths=None):
        """Prepara todos los datos para entrenamiento"""
        print("=== PREPARANDO DATOS PARA ENTRENAMIENTO ===")
        
        # Cargar datos
        if csv_paths:
            # Cargar CSVs específicos
            all_data = []
            for path in csv_paths:
                df = self.load_csv_data(path)
                asset_name = os.path.basename(path).replace('.csv', '')
                df['asset'] = asset_name
                all_data.append(df)
            df = pd.concat(all_data, ignore_index=True)
        else:
            # Cargar todos los activos
            df = self.load_all_assets_data()
        
        # Crear features
        df = self.feature_engineer.create_all_features(df)
        
        # Crear target
        df = self.create_target_variable(df)
        
        # Definir columnas de features (excluir metadatos)
        exclude_cols = ['timestamp', 'asset', 'tipo', 'target', 'target_binary', 
                       'future_close', 'price_change']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Columnas de features: {len(self.feature_columns)}")
        
        # Crear secuencias
        X, y = self.create_sequences(df)
        
        return X, y, df
    
    def split_data_temporal(self, X, y, df):
        """División temporal de datos (no aleatoria)"""
        print("Dividiendo datos temporalmente...")
        
        # Obtener timestamps únicos ordenados
        unique_timestamps = sorted(df['timestamp'].unique())
        n_timestamps = len(unique_timestamps)
        
        # Calcular puntos de corte
        train_end = int(n_timestamps * self.config.TRAIN_SIZE)
        val_end = int(n_timestamps * (self.config.TRAIN_SIZE + self.config.VAL_SIZE))
        
        train_timestamps = unique_timestamps[:train_end]
        val_timestamps = unique_timestamps[train_end:val_end]
        test_timestamps = unique_timestamps[val_end:]
        
        # Crear máscaras basadas en timestamps
        train_mask = df['timestamp'].isin(train_timestamps)
        val_mask = df['timestamp'].isin(val_timestamps)
        test_mask = df['timestamp'].isin(test_timestamps)
        
        # Aplicar máscaras (ajustar por ventana de lookback)
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        # Filtrar índices válidos para secuencias
        valid_train_indices = [i - self.config.LOOKBACK_WINDOW 
                              for i in train_indices 
                              if i >= self.config.LOOKBACK_WINDOW and i < len(X) + self.config.LOOKBACK_WINDOW]
        
        valid_val_indices = [i - self.config.LOOKBACK_WINDOW 
                            for i in val_indices 
                            if i >= self.config.LOOKBACK_WINDOW and i < len(X) + self.config.LOOKBACK_WINDOW]
        
        valid_test_indices = [i - self.config.LOOKBACK_WINDOW 
                             for i in test_indices 
                             if i >= self.config.LOOKBACK_WINDOW and i < len(X) + self.config.LOOKBACK_WINDOW]
        
        # Dividir datos
        X_train = X[valid_train_indices]
        y_train = y[valid_train_indices]
        
        X_val = X[valid_val_indices]
        y_val = y[valid_val_indices]
        
        X_test = X[valid_test_indices]
        y_test = y[valid_test_indices]
        
        print(f"División temporal:")
        print(f"  Entrenamiento: {len(X_train)} muestras")
        print(f"  Validación: {len(X_val)} muestras")
        print(f"  Test: {len(X_test)} muestras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Escala las features usando StandardScaler"""
        print("Escalando features...")
        
        # Entrenar scaler solo con datos de entrenamiento
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Guardar scaler para uso posterior
        scaler_path = os.path.join(self.config.MODELS_DIR, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler guardado en: {scaler_path}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def process_all_data(self, csv_paths=None):
        """Pipeline completo de procesamiento de datos"""
        print("=== PIPELINE COMPLETO DE PROCESAMIENTO ===")
        
        # Crear directorios
        self.config.create_directories()
        
        # Preparar datos
        X, y, df = self.prepare_data_for_training(csv_paths)
        
        # División temporal
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_temporal(X, y, df)
        
        # Escalado
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
        
        # Guardar información de features
        feature_info = {
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns),
            'lookback_window': self.config.LOOKBACK_WINDOW
        }
        joblib.dump(feature_info, os.path.join(self.config.MODELS_DIR, 'feature_info.joblib'))
        
        print("=== PROCESAMIENTO COMPLETADO ===")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled, 
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'raw_df': df
        }

if __name__ == "__main__":
    # Ejemplo de uso
    processor = DataProcessor()
    
    # Procesar todos los datos
    data = processor.process_all_data()
    
    print("Datos procesados y listos para entrenamiento!")