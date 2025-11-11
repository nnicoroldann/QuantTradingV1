# feature_engineering.py - Creación de indicadores técnicos

import pandas as pd
import numpy as np
from config import MLConfig

class FeatureEngineer:
    def __init__(self):
        self.config = MLConfig()
    
    def calculate_sma(self, prices, period):
        """Calcula Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    def calculate_ema(self, prices, period):
        """Calcula Exponential Moving Average"""
        return prices.ewm(span=period).mean()
    
    def calculate_rsi(self, prices, period=14):
        """Calcula Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd, signal)
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, prices, period=20, std=2):
        """Calcula Bollinger Bands"""
        sma = self.calculate_sma(prices, period)
        std_dev = prices.rolling(window=period).std()
        bb_upper = sma + (std_dev * std)
        bb_lower = sma - (std_dev * std)
        bb_width = bb_upper - bb_lower
        bb_position = (prices - bb_lower) / bb_width
        return bb_upper, bb_lower, bb_width, bb_position
    
    def calculate_volatility(self, prices, period=20):
        """Calcula volatilidad histórica"""
        returns = prices.pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(period)
        return volatility
    
    def calculate_volume_features(self, df):
        """Calcula features basadas en volumen"""
        df['volume_sma_10'] = self.calculate_sma(df['volume'], 10)
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        df['price_volume'] = df['close'] * df['volume']
        return df
    
    def calculate_candle_features(self, df):
        """Calcula features de las velas"""
        # Tamaños de cuerpo y sombras
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Ratios
        df['body_to_range'] = df['body_size'] / df['total_range']
        df['upper_shadow_to_body'] = df['upper_shadow'] / (df['body_size'] + 1e-8)
        df['lower_shadow_to_body'] = df['lower_shadow'] / (df['body_size'] + 1e-8)
        
        # Tipo de vela
        df['is_green'] = (df['close'] > df['open']).astype(int)
        df['is_doji'] = (df['body_size'] < df['total_range'] * 0.1).astype(int)
        
        return df
    
    def calculate_momentum_features(self, df):
        """Calcula features de momentum"""
        # Cambios de precio
        for period in [1, 2, 3, 5]:
            df[f'price_change_{period}'] = df['close'].pct_change(period)
            df[f'high_change_{period}'] = df['high'].pct_change(period)
            df[f'low_change_{period}'] = df['low'].pct_change(period)
        
        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        return df
    
    def calculate_time_features(self, df):
        """Calcula features temporales"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)
        
        return df
    
    def create_all_features(self, df):
        """Crea todas las features para el dataset"""
        print(f"Creando features para {len(df)} filas...")
        
        # Hacer copia para no modificar original
        df = df.copy()
        
        # Features temporales
        df = self.calculate_time_features(df)
        
        # Verificar si ya tenemos indicadores calculados
        existing_indicators = ['rsi', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 
                             'adx', 'di_plus', 'di_minus', 'volume_sma', 'volume_ratio']
        
        has_existing_indicators = all(col in df.columns for col in existing_indicators)
        
        if has_existing_indicators:
            print("Usando indicadores ya calculados del CSV...")
            
            # Usar indicadores existentes y crear features adicionales
            # RSI features basadas en el RSI existente
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            
            # Bollinger Bands features usando los existentes
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
            
            # ADX features
            df['adx_strong_trend'] = (df['adx'] > 25).astype(int)
            df['di_bullish'] = (df['di_plus'] > df['di_minus']).astype(int)
            
            # Agregar algunas medias móviles rápidas
            for period in [5, 10]:
                df[f'sma_{period}'] = self.calculate_sma(df['close'], period)
                df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            
            # MACD (calcularlo porque no está en tu CSV)
            macd, macd_signal, macd_hist = self.calculate_macd(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
            
        else:
            print("Calculando todos los indicadores desde cero...")
            
            # Código original para calcular todo
            # Medias móviles
            for period in self.config.SMA_PERIODS:
                df[f'sma_{period}'] = self.calculate_sma(df['close'], period)
                df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            
            for period in self.config.EMA_PERIODS:
                df[f'ema_{period}'] = self.calculate_ema(df['close'], period)
                df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1
            
            # RSI
            df['rsi'] = self.calculate_rsi(df['close'], self.config.RSI_PERIOD)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            
            # MACD
            macd, macd_signal, macd_hist = self.calculate_macd(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_width, bb_position = self.calculate_bollinger_bands(df['close'])
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            df['bb_width'] = bb_width
            df['bb_position'] = bb_position
            df['bb_squeeze'] = (bb_width < bb_width.rolling(20).mean()).astype(int)
            
            # Volatilidad
            df['volatility'] = self.calculate_volatility(df['close'])
            
            # Features de volumen
            df = self.calculate_volume_features(df)
        
        # Estas features siempre las calculamos (son rápidas)
        df = self.calculate_candle_features(df)
        df = self.calculate_momentum_features(df)
        
        # Eliminar filas con NaN (por ventanas de cálculo)
        df = df.dropna()
        
        print(f"Features creadas. Dataset final: {len(df)} filas, {len(df.columns)} columnas")
        
        return df
    
    def get_feature_columns(self):
        """Retorna lista de columnas que son features (no incluye timestamp, target, etc.)"""
        exclude_cols = ['timestamp', 'tipo', 'target', 'target_binary']
        # Esta lista se actualizará después de crear las features
        return None