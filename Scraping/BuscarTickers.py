from .ScrapingWebYF import conexion_pagina_yahoofinance


def buscar_tickers(tickers):
    soup = conexion_pagina_yahoofinance()
    if not soup:
        print("❌ No se pudo establecer conexión")
        return []

    encontrados = []
    for ticker in tickers:
        elementos = soup.find_all(text=lambda text: text and ticker.upper() in str(text).upper())
        if elementos:
            print(f"✅ {ticker}: ENCONTRADO")
            encontrados.append(ticker)
        else:
            print(f"❌ {ticker}: NO ENCONTRADO")
    return encontrados