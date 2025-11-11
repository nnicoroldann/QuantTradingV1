# utils.py - Funciones auxiliares para el sistema ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import joblib
from config import MLConfig

class MLUtils:
    def __init__(self):
        self.config = MLConfig()
    
    def plot_price_and_predictions(self, df, predictions, asset_name="Asset"):
        """Crea gráfico de precios con predicciones"""
        plt.figure(figsize=(15, 8))
        
        # Precio de cierre
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio de Cierre', linewidth=1)
        
        # Marcar predicciones
        if 'prediction' in df.columns:
            buy_signals = df[df['prediction'] == 1]
            sell_signals = df[df['prediction'] == 0]
            
            plt.scatter(buy_signals['timestamp'], buy_signals['close'], 
                       color='green', marker='^', s=50, label='Señal Compra', alpha=0.7)
            plt.scatter(sell_signals['timestamp'], sell_signals['close'], 
                       color='red', marker='v', s=50, label='Señal Venta', alpha=0.7)
        
        plt.title(f'{asset_name} - Precio y Señales de Trading')
        plt.ylabel('Precio ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Volumen
        plt.subplot(2, 1, 2)
        plt.bar(df['timestamp'], df['volume'], alpha=0.6, color='blue')
        plt.title('Volumen')
        plt.ylabel('Volumen')
        plt.xlabel('Tiempo')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt
    
    def plot_feature_importance(self, feature_importance_df, top_n=20):
        """Gráfico de importancia de features"""
        plt.figure(figsize=(12, 8))
        
        top_features = feature_importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importancia')
        plt.title(f'Top {top_n} Features Más Importantes')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt
    
    def plot_model_comparison(self, results_dict):
        """Compara métricas de diferentes modelos"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'win_rate']
        models = list(results_dict.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            train_values = [results_dict[model]['train_metrics'][metric] for model in models]
            val_values = [results_dict[model]['val_metrics'][metric] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[i].bar(x - width/2, train_values, width, label='Train', alpha=0.7)
            axes[i].bar(x + width/2, val_values, width, label='Validation', alpha=0.7)
            
            axes[i].set_xlabel('Modelos')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'Comparación {metric.capitalize()}')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(models, rotation=45)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Ocultar el último subplot si no se usa
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        return plt
    
    def plot_prediction_distribution(self, y_true, y_pred, y_proba):
        """Análisis de la distribución de predicciones"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Distribución de clases reales
        axes[0, 0].hist(y_true, bins=2, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Distribución Clases Reales')
        axes[0, 0].set_xlabel('Clase (0=Bajada, 1=Subida)')
        axes[0, 0].set_ylabel('Frecuencia')
        
        # Distribución de predicciones
        axes[0, 1].hist(y_pred, bins=2, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('Distribución Predicciones')
        axes[0, 1].set_xlabel('Predicción (0=Bajada, 1=Subida)')
        axes[0, 1].set_ylabel('Frecuencia')
        
        # Distribución de probabilidades
        axes[1, 0].hist(y_proba, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title('Distribución Probabilidades')
        axes[1, 0].set_xlabel('Probabilidad de Subida')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].axvline(x=0.5, color='red', linestyle='--', label='Umbral 50%')
        axes[1, 0].legend()
        
        # Matriz de confusión
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Matriz de Confusión')
        axes[1, 1].set_xlabel('Predicción')
        axes[1, 1].set_ylabel('Real')
        
        plt.tight_layout()
        return plt
    
    def calculate_backtest_metrics(self, df_with_predictions):
        """Calcula métricas de backtesting"""
        df = df_with_predictions.copy()
        
        # Simular trading simple
        df['position'] = 0  # 0 = sin posición, 1 = largo, -1 = corto
        df['returns'] = df['close'].pct_change()
        
        # Generar señales de posición
        df.loc[df['prediction'] == 1, 'position'] = 1  # Largo cuando predice subida
        df.loc[df['prediction'] == 0, 'position'] = -1  # Corto cuando predice bajada
        
        # Calcular retornos de estrategia
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        df['strategy_returns'] = df['strategy_returns'].fillna(0)
        
        # Retornos acumulados
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()
        
        # Métricas
        total_return = df['cumulative_strategy_returns'].iloc[-1] - 1
        buy_hold_return = df['cumulative_returns'].iloc[-1] - 1
        
        # Volatilidad anualizada (asumiendo datos cada 2 horas)
        periods_per_year = 365 * 24 / 2  # 4380 períodos por año
        volatility = df['strategy_returns'].std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        sharpe = (df['strategy_returns'].mean() * periods_per_year) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        running_max = df['cumulative_strategy_returns'].expanding().max()
        drawdown = (df['cumulative_strategy_returns'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (df['strategy_returns'] > 0).sum()
        total_trades = (df['strategy_returns'] != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades
        }
        
        return metrics, df
    
    def plot_backtest_results(self, df_backtest, asset_name="Asset"):
        """Gráfico de resultados del backtesting"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Retornos acumulados
        axes[0].plot(df_backtest['timestamp'], df_backtest['cumulative_returns'], 
                    label='Buy & Hold', linewidth=2)
        axes[0].plot(df_backtest['timestamp'], df_backtest['cumulative_strategy_returns'], 
                    label='Estrategia ML', linewidth=2)
        axes[0].set_title(f'{asset_name} - Retornos Acumulados')
        axes[0].set_ylabel('Retorno Acumulado')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        running_max = df_backtest['cumulative_strategy_returns'].expanding().max()
        drawdown = (df_backtest['cumulative_strategy_returns'] - running_max) / running_max
        
        axes[1].fill_between(df_backtest['timestamp'], drawdown, 0, alpha=0.3, color='red')
        axes[1].plot(df_backtest['timestamp'], drawdown, color='red', linewidth=1)
        axes[1].set_title('Drawdown de la Estrategia')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Retornos diarios
        axes[2].bar(df_backtest['timestamp'], df_backtest['strategy_returns'], 
                   alpha=0.6, color=['green' if x > 0 else 'red' for x in df_backtest['strategy_returns']])
        axes[2].set_title('Retornos por Período')
        axes[2].set_ylabel('Retorno (%)')
        axes[2].set_xlabel('Tiempo')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt
    
    def create_trading_report(self, backtest_metrics, asset_name="Asset"):
        """Crea reporte de trading en texto"""
        report = f"""
=== REPORTE DE BACKTESTING - {asset_name.upper()} ===

RETORNOS:
  Retorno Total Estrategia: {backtest_metrics['total_return']:.2%}
  Retorno Buy & Hold: {backtest_metrics['buy_hold_return']:.2%}
  Retorno Excesivo: {backtest_metrics['excess_return']:.2%}

RIESGO:
  Volatilidad Anualizada: {backtest_metrics['volatility']:.2%}
  Máximo Drawdown: {backtest_metrics['max_drawdown']:.2%}
  Ratio de Sharpe: {backtest_metrics['sharpe_ratio']:.3f}

TRADING:
  Total de Operaciones: {backtest_metrics['total_trades']}
  Operaciones Ganadoras: {backtest_metrics['winning_trades']}
  Win Rate: {backtest_metrics['win_rate']:.2%}

EVALUACIÓN:
  {'✅ Estrategia SUPERÓ buy & hold' if backtest_metrics['excess_return'] > 0 else '❌ Estrategia NO superó buy & hold'}
  {'✅ Sharpe ratio ACEPTABLE' if backtest_metrics['sharpe_ratio'] > 1 else '⚠️ Sharpe ratio BAJO' if backtest_metrics['sharpe_ratio'] > 0 else '❌ Sharpe ratio NEGATIVO'}
  {'✅ Win rate BUENO' if backtest_metrics['win_rate'] > 0.55 else '⚠️ Win rate MODERADO' if backtest_metrics['win_rate'] > 0.45 else '❌ Win rate BAJO'}
        """
        
        return report
    
    def validate_data_quality(self, df):
        """Valida la calidad de los datos"""
        issues = []
        
        # Verificar datos faltantes
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            issues.append(f"Datos faltantes encontrados: {missing_data[missing_data > 0].to_dict()}")
        
        # Verificar duplicados de timestamp
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Timestamps duplicados: {duplicates}")
        
        # Verificar orden cronológico
        df_sorted = df.sort_values('timestamp')
        if not df['timestamp'].equals(df_sorted['timestamp']):
            issues.append("Datos no están en orden cronológico")
        
        # Verificar precios negativos o cero
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                invalid_prices = (df[col] <= 0).sum()
                if invalid_prices > 0:
                    issues.append(f"Precios inválidos en {col}: {invalid_prices}")
        
        # Verificar coherencia de precios OHLC
        if all(col in df.columns for col in price_cols):
            # High debe ser >= max(open, close)
            high_issues = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
            if high_issues > 0:
                issues.append(f"High menor que open/close: {high_issues} casos")
            
            # Low debe ser <= min(open, close)
            low_issues = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
            if low_issues > 0:
                issues.append(f"Low mayor que open/close: {low_issues} casos")
        
        # Verificar volumen negativo
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Volumen negativo: {negative_volume} casos")
        
        return issues
    
    def prepare_data_summary(self, df):
        """Crea resumen de los datos"""
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max() if 'timestamp' in df.columns else None
            },
            'price_stats': {},
            'volume_stats': {}
        }
        
        # Estadísticas de precios
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                summary['price_stats'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
        
        # Estadísticas de volumen
        if 'volume' in df.columns:
            summary['volume_stats'] = {
                'min': df['volume'].min(),
                'max': df['volume'].max(),
                'mean': df['volume'].mean(),
                'std': df['volume'].std()
            }
        
        return summary
    
    def save_plots_to_file(self, plt_object, filename, dpi=300):
        """Guarda gráficos en archivo"""
        filepath = os.path.join(self.config.RESULTS_DIR, filename)
        plt_object.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Gráfico guardado en: {filepath}")
        return filepath

def check_ml_system_status():
    """Verifica el estado del sistema ML"""
    config = MLConfig()
    status = {
        'directories': {},
        'models': {},
        'data': {}
    }
    
    # Verificar directorios
    directories = [config.DATA_DIR, config.MODELS_DIR, config.RESULTS_DIR]
    for directory in directories:
        status['directories'][directory] = os.path.exists(directory)
    
    # Verificar modelos
    model_files = ['latest_model.joblib', 'scaler.joblib', 'feature_info.joblib', 'model_metadata.joblib']
    for model_file in model_files:
        filepath = os.path.join(config.MODELS_DIR, model_file)
        status['models'][model_file] = os.path.exists(filepath)
    
    # Verificar datos
    for asset in config.ASSETS:
        csv_path = os.path.join(config.DATA_DIR, f"{asset}.csv")
        status['data'][f"{asset}.csv"] = os.path.exists(csv_path)
    
    return status

def print_system_status():
    """Imprime el estado del sistema"""
    status = check_ml_system_status()
    
    print("=== ESTADO DEL SISTEMA ML ===")
    
    print("\nDIRECTORIOS:")
    for directory, exists in status['directories'].items():
        icon = "✅" if exists else "❌"
        print(f"  {icon} {directory}")
    
    print("\nMODELOS:")
    for model_file, exists in status['models'].items():
        icon = "✅" if exists else "❌"
        print(f"  {icon} {model_file}")
    
    print("\nDATOS:")
    for data_file, exists in status['data'].items():
        icon = "✅" if exists else "❌"
        print(f"  {icon} {data_file}")
    
    # Determinar si el sistema está listo
    all_models_exist = all(status['models'].values())
    some_data_exists = any(status['data'].values())
    
    print(f"\n{'✅ SISTEMA LISTO' if all_models_exist and some_data_exists else '❌ SISTEMA NO ESTÁ LISTO'}")
    
    if not all_models_exist:
        print("  ⚠️  Necesitas entrenar el modelo primero (ejecutar model_trainer.py)")
    
    if not some_data_exists:
        print("  ⚠️  Necesitas datos CSV en la carpeta data/")

if __name__ == "__main__":
    print_system_status()