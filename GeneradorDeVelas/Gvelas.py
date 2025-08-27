# Gvelas.py
import pandas as pd
from datetime import datetime, timedelta
import random

class GeneradorVelas:
    def __init__(self):
        pass

    def generar_datos_simulados(self, fecha_inicio, fecha_fin, precio_inicial=1000):
        """
        Genera ticks simulados desde fecha_inicio hasta fecha_fin, sin pasar la hora actual
        """
        datos = []
        precio_base = precio_inicial

        timestamp = datetime.strptime(fecha_inicio, "%Y-%m-%d %H:%M:%S")
        fin = min(datetime.strptime(fecha_fin, "%Y-%m-%d %H:%M:%S"), datetime.now())

        while timestamp < fin:
            variacion = random.uniform(-0.02, 0.02)
            precio_base = precio_base * (1 + variacion)
            volumen = random.randint(100, 5000)

            datos.append({
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'precio': round(precio_base, 2),
                'volumen': volumen
            })

            timestamp += timedelta(minutes=random.randint(1,3))

        return datos


    def procesar_datos_en_velas(self, datos_raw, intervalo_minutos=10):
        if not datos_raw:
            return []

        df = pd.DataFrame(datos_raw)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        intervalo_str = f'{intervalo_minutos}min'
        df['intervalo'] = df['timestamp'].dt.floor(intervalo_str)

        velas = []
        for intervalo, grupo in df.groupby('intervalo'):
            grupo_ordenado = grupo.sort_values('timestamp')
            vela = {
                'timestamp': intervalo.strftime("%Y-%m-%d %H:%M:%S"),
                'open': float(grupo_ordenado.iloc[0]['precio']),
                'high': float(grupo_ordenado['precio'].max()),
                'low': float(grupo_ordenado['precio'].min()),
                'close': float(grupo_ordenado.iloc[-1]['precio']),
                'volume': int(grupo_ordenado['volumen'].sum()),
                'num_trades': len(grupo_ordenado),
                'tipo': 'alcista' if grupo_ordenado.iloc[-1]['precio'] >= grupo_ordenado.iloc[0]['precio'] else 'bajista'
            }
            velas.append(vela)

        return sorted(velas, key=lambda x: x['timestamp'])
