# app.py - API Web para monitorear tu aplicación
import os
import pandas as pd
from flask import Flask, jsonify, render_template_string
from datetime import datetime, timedelta
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Monitor de Tickers - Análisis en Tiempo Real</title>
        <meta http-equiv="refresh" content="30">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
            .stats { display: flex; gap: 20px; margin: 20px 0; }
            .stat-card { background: #f8f9fa; padding: 15px; border-radius: 8px; flex: 1; text-align: center; }
            .stat-number { font-size: 24px; font-weight: bold; color: #667eea; }
            .file-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }
            .file-card { background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 8px; }
            .api-list { background: #f8f9fa; padding: 20px; border-radius: 8px; }
            .api-list a { color: #667eea; text-decoration: none; }
            .api-list a:hover { text-decoration: underline; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #28a745; margin-right: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Monitor de Análisis de Tickers</h1>
                <p><span class="status-indicator"></span>Sistema Activo - Última actualización: {{ timestamp }}</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{{ files|length }}</div>
                    <div>Tickers Monitoreados</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ total_candles }}</div>
                    <div>Total Velas Generadas</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">24/7</div>
                    <div>Monitoreo Continuo</div>
                </div>
            </div>

            <h3>📊 Archivos de Datos Generados:</h3>
            <div class="file-list">
            {% for file_info in file_details %}
                <div class="file-card">
                    <h4><a href="/ticker/{{ file_info.ticker }}">{{ file_info.filename }}</a></h4>
                    <p><strong>Ticker:</strong> {{ file_info.ticker }}</p>
                    <p><strong>Velas:</strong> {{ file_info.candle_count }}</p>
                    <p><strong>Última actualización:</strong> {{ file_info.last_update }}</p>
                </div>
            {% endfor %}
            </div>
            
            <div class="api-list">
                <h3>🔗 APIs Disponibles:</h3>
                <ul>
                    <li><a href="/status" target="_blank">/status</a> - Estado del sistema (JSON)</li>
                    <li><a href="/files" target="_blank">/files</a> - Lista de archivos (JSON)</li>
                    <li><a href="/ticker/GGAL" target="_blank">/ticker/TICKER</a> - Datos de un ticker específico</li>
                    <li><a href="/indicators/GGAL" target="_blank">/indicators/TICKER</a> - Indicadores técnicos</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    ''', 
    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
    files=get_output_files(),
    file_details=get_file_details(),
    total_candles=get_total_candles())

@app.route('/status')
def status():
    output_dir = "output"
    files = get_output_files()
    
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "files_count": len(files),
        "files": files,
        "output_directory": output_dir
    })

@app.route('/files')
def list_files():
    return jsonify(get_output_files())

@app.route('/ticker/<ticker_name>')
def get_ticker_data(ticker_name):
    try:
        file_path = f"output/velas_{ticker_name}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Obtener las últimas 50 velas
            latest_data = df.tail(50).to_dict('records')
            
            return jsonify({
                "ticker": ticker_name,
                "total_candles": len(df),
                "latest_data": latest_data,
                "last_update": df['timestamp'].iloc[-1] if not df.empty else None
            })
        else:
            return jsonify({"error": f"No se encontró archivo para {ticker_name}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/indicators/<ticker_name>')
def get_ticker_indicators(ticker_name):
    try:
        indicators_file = f"output/indicadores_{ticker_name}.csv"
        if os.path.exists(indicators_file):
            df = pd.read_csv(indicators_file)
            latest_indicators = df.tail(10).to_dict('records')
            
            return jsonify({
                "ticker": ticker_name,
                "indicators": latest_indicators,
                "columns": list(df.columns),
                "total_records": len(df)
            })
        else:
            return jsonify({"error": f"No se encontraron indicadores para {ticker_name}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_file_details():
    details = []
    output_dir = "output"
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.startswith('velas_') and file.endswith('.csv'):
                ticker = file.replace('velas_', '').replace('.csv', '')
                file_path = os.path.join(output_dir, file)
                
                try:
                    df = pd.read_csv(file_path)
                    candle_count = len(df)
                    last_update = df['timestamp'].iloc[-1] if not df.empty else "N/A"
                except:
                    candle_count = 0
                    last_update = "Error al leer"
                
                details.append({
                    'filename': file,
                    'ticker': ticker,
                    'candle_count': candle_count,
                    'last_update': last_update
                })
    return details

def get_total_candles():
    total = 0
    output_dir = "output"
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.startswith('velas_') and file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(output_dir, file))
                    total += len(df)
                except:
                    continue
    return total

def get_output_files():
    output_dir = "output"
    if os.path.exists(output_dir):
        return [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    return []

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)