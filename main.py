# main.py
import os
import time
from datetime import datetime, timedelta
import pandas as pd
from Scraping.BuscarTickers import buscar_tickers
from GeneradorDeVelas.Gvelas import GeneradorVelas
from Indicadores.IndicadoresT import analyze_candles, save_indicators_to_csv

def generar_velas_historicas_y_continuas(tickers, dias_historicos=7, horas_intervalo=2, intervalo_vela_min=10, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    generador = GeneradorVelas()
    
    # Obtener fecha inicial histórica
    fecha_inicio_historica = datetime.now() - timedelta(days=dias_historicos)
    ahora = datetime.now()
    
    encontrados = buscar_tickers(tickers)
    if not encontrados:
        print("⚠️ No se encontraron tickers válidos.")
        return
    
    # 1️⃣ Generar velas históricas
    for ticker in encontrados:
        print(f"\n📊 Generando velas históricas para {ticker}...")
        csv_path = os.path.join(output_dir, f"velas_{ticker}.csv")
        
        if os.path.exists(csv_path):
            df_existente = pd.read_csv(csv_path)
            ultima_fecha = pd.to_datetime(df_existente['timestamp']).max()
            fecha_inicio = ultima_fecha + timedelta(seconds=1)
            precio_base = df_existente['close'].iloc[-1]
        else:
            fecha_inicio = fecha_inicio_historica
            precio_base = 1000
        
        fecha_fin = ahora
        horas_totales = (fecha_fin - fecha_inicio).total_seconds() / 3600
        
        datos_tick = generador.generar_datos_simulados(
            fecha_inicio.strftime("%Y-%m-%d %H:%M:%S"),
            fecha_fin.strftime("%Y-%m-%d %H:%M:%S"),
            precio_inicial=precio_base
        )
        
        velas = generador.procesar_datos_en_velas(datos_tick, intervalo_minutos=intervalo_vela_min)
        
        if velas:
            df_nuevo = pd.DataFrame(velas)
            
            if os.path.exists(csv_path):
                df_combinado = pd.concat([df_existente, df_nuevo]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                df_combinado.to_csv(csv_path, index=False)
            else:
                df_nuevo.to_csv(csv_path, index=False)
            
            print(f"✅ Velas históricas de {ticker} guardadas en {csv_path}")
            
            # 🔥 NUEVO: Aplicar indicadores técnicos
            try:
                df_final = pd.read_csv(csv_path)
                df_con_indicadores, analysis = analyze_candles(df_final, ticker)
                save_indicators_to_csv(df_con_indicadores, ticker, output_dir)
            except Exception as e:
                print(f"❌ Error aplicando indicadores a {ticker}: {e}")
                
        else:
            print("⚠️ No se generaron velas históricas para este ticker.")
    
    # 2️⃣ Loop continuo cada 2 horas
    while True:
        ahora = datetime.now()
        print(f"\n⏱ Inicio de generación de velas a las {ahora.strftime('%Y-%m-%d %H:%M:%S')}")
        
        for ticker in encontrados:
            print(f"\n📊 Procesando ticker {ticker}...")
            csv_path = os.path.join(output_dir, f"velas_{ticker}.csv")
            
            df_existente = pd.read_csv(csv_path)
            ultima_fecha = pd.to_datetime(df_existente['timestamp']).max()
            fecha_inicio = ultima_fecha + timedelta(seconds=1)
            fecha_fin = fecha_inicio + timedelta(hours=horas_intervalo)
            precio_base = df_existente['close'].iloc[-1]
            
            datos_tick = generador.generar_datos_simulados(
                fecha_inicio.strftime("%Y-%m-%d %H:%M:%S"),
                fecha_fin.strftime("%Y-%m-%d %H:%M:%S"),
                precio_inicial=precio_base
            )
            
            velas = generador.procesar_datos_en_velas(datos_tick, intervalo_minutos=intervalo_vela_min)
            
            if velas:
                df_nuevo = pd.DataFrame(velas)
                df_combinado = pd.concat([df_existente, df_nuevo]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                df_combinado.to_csv(csv_path, index=False)
                
                print(f"✅ Nuevas velas de {ticker} guardadas en {csv_path}")
                
                # 🔥 NUEVO: Actualizar indicadores técnicos
                try:
                    df_con_indicadores, analysis = analyze_candles(df_combinado, ticker)
                    save_indicators_to_csv(df_con_indicadores, ticker, output_dir)
                except Exception as e:
                    print(f"❌ Error actualizando indicadores de {ticker}: {e}")
                    
            else:
                print("⚠️ No se generaron nuevas velas para este ticker.")
        
        print(f"\n⏳ Esperando {horas_intervalo} horas para el próximo ciclo...\n")
        time.sleep(horas_intervalo * 3600)

if __name__ == "__main__":
    tickers = ["GGAL.BA", "VIST.BA", "YPFD.BA"]
    generar_velas_historicas_y_continuas(tickers)