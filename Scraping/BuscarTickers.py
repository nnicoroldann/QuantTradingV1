import requests
from bs4 import BeautifulSoup
import time

def conexion_pagina_yahoofinance():
    """Conexión base a Yahoo Finance"""
    url = "https://finance.yahoo.com"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        print("Conectando a Yahoo Finance...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        print("✅ Conexión exitosa!")
        return soup
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión: {e}")
        return None

def verificar_ticker_yahoo_api(ticker):
    """Verifica ticker usando la API de Yahoo Finance"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                # Verificar que hay datos válidos
                result = data['chart']['result'][0]
                if result and 'meta' in result:
                    return True
        return False
    except Exception as e:
        print(f"❌ Error verificando {ticker}: {e}")
        return False

def buscar_tickers(tickers):
    """Busca tickers usando múltiples métodos"""
    print("🔍 Iniciando búsqueda de tickers...")
    encontrados = []
    
    for ticker in tickers:
        print(f"\n🔍 Verificando {ticker}...")
        
        # Método principal: API de Yahoo Finance
        if verificar_ticker_yahoo_api(ticker):
            print(f"✅ {ticker}: ENCONTRADO")
            encontrados.append(ticker)
        else:
            print(f"❌ {ticker}: NO ENCONTRADO")
        
        # Pausa para evitar rate limiting
        time.sleep(0.5)
    
    print(f"\n📊 Resumen: {len(encontrados)}/{len(tickers)} tickers encontrados")
    return encontrados