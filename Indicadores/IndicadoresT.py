# Indicadores/IndicadoresT.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

class TechnicalIndicators:
    def __init__(self, df: pd.DataFrame):
        """
        df debe contener columnas: 'open', 'high', 'low', 'close', 'volume'
        """
        self.df = df.copy()
        
    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        """RSI - Relative Strength Index"""
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        return self.df
    
    def add_bollinger_bands(self, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Bollinger Bands"""
        self.df['bb_middle'] = self.df['close'].rolling(window=period).mean()
        bb_std = self.df['close'].rolling(window=period).std()
        self.df['bb_upper'] = self.df['bb_middle'] + (bb_std * std_dev)
        self.df['bb_lower'] = self.df['bb_middle'] - (bb_std * std_dev)
        self.df['bb_width'] = self.df['bb_upper'] - self.df['bb_lower']
        return self.df
    
    def add_adx(self, period: int = 14) -> pd.DataFrame:
        """ADX - Average Directional Index"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = low.diff() * -1
        
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        dm_plus[(dm_plus - dm_minus) < 0] = 0
        dm_minus[(dm_minus - dm_plus) < 0] = 0
        
        # Smoothed values
        atr = tr.rolling(window=period).mean()
        di_plus = (dm_plus.rolling(window=period).mean() / atr) * 100
        di_minus = (dm_minus.rolling(window=period).mean() / atr) * 100
        
        # ADX calculation
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        self.df['adx'] = dx.rolling(window=period).mean()
        self.df['di_plus'] = di_plus
        self.df['di_minus'] = di_minus
        
        return self.df
    
    def add_volume_profile(self, bins: int = 20) -> Dict:
        """Volume Profile Analysis"""
        if 'volume' not in self.df.columns:
            print("⚠️ No hay columna 'volume', creando volumen simulado...")
            self.df['volume'] = np.random.randint(1000, 10000, len(self.df))
        
        price_range = self.df['high'].max() - self.df['low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        
        for i in range(bins):
            price_level = self.df['low'].min() + (i * bin_size)
            price_level_upper = price_level + bin_size
            
            # Volumen en este rango de precio
            mask = ((self.df['low'] <= price_level_upper) & 
                   (self.df['high'] >= price_level))
            volume_at_level = self.df.loc[mask, 'volume'].sum()
            
            volume_profile[round(price_level, 2)] = volume_at_level
        
        # Encuentra POC (Point of Control) - precio con mayor volumen
        poc_price = max(volume_profile, key=volume_profile.get) if volume_profile else 0
        
        # Añade métricas al dataframe
        self.df['volume_sma'] = self.df['volume'].rolling(window=20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma']
        
        return {
            'volume_profile': volume_profile,
            'poc_price': poc_price,
            'total_volume': self.df['volume'].sum()
        }
    
    def add_fibonacci_retracements(self) -> Dict:
        """Fibonacci Retracements Automático"""
        # Encuentra swing high y swing low más recientes
        window = min(10, len(self.df) // 4)  # Ajustar ventana según datos disponibles
        
        if len(self.df) < window * 2:
            # Si hay pocos datos, usar min/max globales
            swing_high = self.df['high'].max()
            swing_low = self.df['low'].min()
            trend_direction = 'up' if self.df['close'].iloc[-1] > self.df['close'].iloc[0] else 'down'
        else:
            # Identifica peaks y valleys
            highs = self.df['high'].rolling(window=window, center=True).max() == self.df['high']
            lows = self.df['low'].rolling(window=window, center=True).min() == self.df['low']
            
            # Últimos swing points
            last_high_idx = self.df[highs].index[-1] if highs.any() else self.df['high'].idxmax()
            last_low_idx = self.df[lows].index[-1] if lows.any() else self.df['low'].idxmin()
            
            swing_high = self.df.loc[last_high_idx, 'high']
            swing_low = self.df.loc[last_low_idx, 'low']
            trend_direction = 'up' if last_high_idx > last_low_idx else 'down'
        
        # Niveles de Fibonacci
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        fibonacci_prices = {}
        
        price_diff = swing_high - swing_low
        
        for level in fib_levels:
            if trend_direction == 'up':  # Uptrend retracement
                fib_price = swing_high - (price_diff * level)
            else:  # Downtrend retracement
                fib_price = swing_low + (price_diff * level)
            
            fibonacci_prices[f'fib_{level}'] = round(fib_price, 2)
        
        return {
            'fibonacci_levels': fibonacci_prices,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'trend_direction': trend_direction
        }
    
    def generate_all_indicators(self) -> Tuple[pd.DataFrame, Dict]:
        """Genera todos los indicadores de una vez"""
        try:
            # Añadir indicadores al dataframe
            self.add_rsi()
            self.add_bollinger_bands()
            self.add_adx()
            
            # Análisis adicionales
            volume_analysis = self.add_volume_profile()
            fibonacci_analysis = self.add_fibonacci_retracements()
            
            analysis_data = {
                'volume_analysis': volume_analysis,
                'fibonacci_analysis': fibonacci_analysis
            }
            
            return self.df, analysis_data
        except Exception as e:
            print(f"❌ Error generando indicadores: {e}")
            return self.df, {}

def analyze_candles(df: pd.DataFrame, ticker: str = ""):
    """
    Función principal para analizar velas con todos los indicadores
    """
    if len(df) < 20:
        print(f"⚠️ {ticker}: Datos insuficientes para análisis completo ({len(df)} velas)")
        return df, {}
    
    try:
        indicators = TechnicalIndicators(df)
        df_with_indicators, analysis = indicators.generate_all_indicators()
        
        # Resumen del análisis
        current_rsi = df_with_indicators['rsi'].iloc[-1] if not df_with_indicators['rsi'].isna().all() else 50
        current_adx = df_with_indicators['adx'].iloc[-1] if not df_with_indicators['adx'].isna().all() else 20
        
        print(f"\n📈 ANÁLISIS TÉCNICO - {ticker}")
        print(f"RSI: {current_rsi:.2f}", end=" ")
        if current_rsi > 70:
            print("(Sobrecomprado)")
        elif current_rsi < 30:
            print("(Sobrevendido)")
        else:
            print("(Neutral)")
        
        print(f"ADX: {current_adx:.2f}", end=" ")
        if current_adx > 25:
            print("(Tendencia fuerte)")
        else:
            print("(Tendencia débil)")
        
        if analysis.get('volume_analysis'):
            print(f"POC Price: {analysis['volume_analysis']['poc_price']}")
        
        if analysis.get('fibonacci_analysis'):
            print(f"Fibonacci: {analysis['fibonacci_analysis']['trend_direction']}")
        
        return df_with_indicators, analysis
        
    except Exception as e:
        print(f"❌ Error en análisis de {ticker}: {e}")
        return df, {}

def save_indicators_to_csv(df: pd.DataFrame, ticker: str, output_dir: str):
    """Guarda el DataFrame con indicadores en un archivo separado"""
    try:
        indicators_dir = os.path.join(output_dir, "indicadores")
        os.makedirs(indicators_dir, exist_ok=True)
        
        indicators_path = os.path.join(indicators_dir, f"indicadores_{ticker}.csv")
        df.to_csv(indicators_path, index=False)
        print(f"📊 Indicadores de {ticker} guardados en {indicators_path}")
        
    except Exception as e:
        print(f"❌ Error guardando indicadores de {ticker}: {e}")