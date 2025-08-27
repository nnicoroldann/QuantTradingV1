import requests
from bs4 import BeautifulSoup

def conexion_pagina_yahoofinance():
    url= "https://finance.yahoo.com"


    #simnular que el scrip viene de una web y no de un auto scrip
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        
        #guardamos todo en el response, en caso de que la peticion 
        # no de estado 200 tira un except
        print("Conectando a Yahoo Finance...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        print("✅ Conexión exitosa!")
        
        return soup
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión: {e}")
        return None
