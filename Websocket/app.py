import asyncio
import json
from datetime import datetime, timedelta
import os
import time
import pytz
import httpx
import numpy as np
import threading
from dotenv import load_dotenv
from supabase import Client, create_client
from quart import Quart, websocket
from quart_cors import cors
from Auth.jwt_factory import verify_jwt
from AI_Modules.Regression import Regression
from AI_Modules.Autoencoder import AutoencoderDetector
from datetime import datetime

load_dotenv()

app = Quart(__name__)
app = cors(app, allow_origin="*")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

mex_tz = pytz.timezone("America/Mexico_City")
active_connections = {"arduino": set(), "react": set()}

# Acumuladores
hourly_time = 0
hourly_consumption = 0
lock = asyncio.Lock()

# Datos climáticos
weather_temps = []
weather_sensaciones = []
weather_precips = []

# ML Models y datos
reg_model = None
autoencoder = None
X_train, y_train = [], []
#consumo_segundo_buffer = []
MAX_TRAIN_SIZE = 500 

last_arduino_log = datetime.now(mex_tz)

last_notif_sent = datetime.min 


def instantiate_models():
    global reg_model, autoencoder
    if reg_model is None:
        reg_model = Regression()
        print("Modelo de Regresión cargado")
    if autoencoder is None:
        autoencoder = AutoencoderDetector(input_dim=2)
        print("Autoencoder cargado")


def destroy_models():
    global reg_model, autoencoder, X_train, y_train
    reg_model = None
    if autoencoder:
        autoencoder.save()
    autoencoder = None
    X_train.clear()
    y_train.clear()
    print("Modelos ML destruidos (React desconectado)")


async def fetch_weather():
   
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                "http://api.weatherapi.com/v1/current.json",
                params={"key": os.getenv("WEATHER_API_KEY"), "q": "zapopan"}
            )
        if response.status_code == 200:
            current = response.json().get("current", {})
            temp = current.get("temp_c", 0)
            feelslike = current.get("feelslike_c", 0)
            precip = current.get("humidity", 0)

            weather_temps[:] = (weather_temps + [temp])[-6:]
            weather_sensaciones[:] = (weather_sensaciones + [feelslike])[-6:]
            weather_precips[:] = (weather_precips + [precip])[-6:]

            print(f"Clima actualizado: temp={temp}°C, sensacion={feelslike}°C, humedad={precip}%")
        else:
            print(f"WeatherAPI error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error fetch_weather: {e}")


async def weather_task():
 
    while True:
        await fetch_weather()
        await asyncio.sleep(600) 


async def save_hourly():
  
    global hourly_time, hourly_consumption
    now_local = datetime.now(mex_tz)
    now_utc = now_local.astimezone(pytz.UTC)
    fecha_ajustada = now_utc - timedelta(hours=6)

    async with lock:
        temp_prom = sum(weather_temps) / len(weather_temps) if weather_temps else 0
        sensacion_prom = sum(weather_sensaciones) / len(weather_sensaciones) if weather_sensaciones else 0
        precip_prom = sum(weather_precips) / len(weather_precips) if weather_precips else 0

        try:
            supabase.table("registrosHora").insert({
                "tiempoActivo": hourly_time,
                "consumo": hourly_consumption,
                "temperaturaProm": temp_prom,
                "sensacionProm": sensacion_prom,
                "humedadProm": precip_prom,
                "hora": now_local.hour,
                "dia": now_local.isoweekday(),
                "mes": now_local.month,
                "personas": 4,
                "sensor": 1,
                "usuario": 1,
                "fecha": fecha_ajustada.isoformat()
            }).execute()
            print(f"Guardado horario {now_local.hour}h: tiempo={hourly_time}, consumo={hourly_consumption}")
        except Exception as e:
            print(f"Error guardando datos horarios: {e}")
        finally:
            hourly_time = 0
            hourly_consumption = 0


async def save_daily():
 
    now_local = datetime.now(mex_tz) - timedelta(days=1)  
    now_utc = now_local.astimezone(pytz.UTC)

    start_of_day_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day_local = start_of_day_local + timedelta(days=1)
    start_utc = start_of_day_local.astimezone(pytz.UTC).isoformat()
    end_utc = end_of_day_local.astimezone(pytz.UTC).isoformat()

    try:
        result = supabase.table("registrosHora").select(
            "tiempoActivo, consumo, temperaturaProm, sensacionProm, humedadProm"
        ).gte("fecha", start_utc).lt("fecha", end_utc).execute()

        rows = result.data
        if rows:
            total_tiempo = sum(r['tiempoActivo'] for r in rows)
            total_consumo = sum(r['consumo'] for r in rows)
            temp_prom = sum(r['temperaturaProm'] for r in rows) / len(rows)
            sensacion_prom = sum(r['sensacionProm'] for r in rows) / len(rows)
            precip_prom = sum(r['humedadProm'] for r in rows) / len(rows)

            supabase.table("registrosDia").insert({
                "tiempoActivo": total_tiempo,
                "consumo": total_consumo,
                "temperaturaProm": temp_prom,
                "sensacionProm": sensacion_prom,
                "humedadProm": precip_prom,
                "dia": now_local.isoweekday(),
                "mes": now_local.month,
                "personas": 4,
                "sensor": 1,
                "usuario": 1,
                "fecha": now_utc.isoformat()
            }).execute()
            print(f"Guardado diario {now_local.date()}")
        else:
            print("No hay registros horarios para el día actual")
    except Exception as e:
        print(f"Error guardando datos diarios: {e}")


async def hourly_task():

    while True:
        now = datetime.now(mex_tz)
        seconds_until_next_hour = (60 - now.minute) * 60 - now.second
        await asyncio.sleep(seconds_until_next_hour)
        await fetch_weather()
        await save_hourly()

"""
async def save_consumo_segundo_task():
    global consumo_segundo_buffer
    while True:
        await asyncio.sleep(10)  
        async with lock:
            if consumo_segundo_buffer:
                try:
                    supabase.table("consumoSegundo").insert(consumo_segundo_buffer).execute()
                    print(f"Guardados {len(consumo_segundo_buffer)} registros en consumoSegundo")
                    consumo_segundo_buffer = [] 
                except Exception as e:
                    print(f"Error guardando consumoSegundo: {e}")

"""

def run_daily_thread():
   
    while True:
        now = datetime.now(mex_tz)
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_until_midnight = (next_midnight - now).total_seconds()
        print(f"Esperando {seconds_until_midnight}s para guardar diario...")
        time.sleep(seconds_until_midnight)
        try:
            asyncio.run(save_daily())
        except Exception as e:
            print(f"Error en run_daily_thread: {e}")


async def ml_training_task():
  
    while True:
        if active_connections["react"]:
            try:
                if reg_model and autoencoder and len(X_train) >= 10:
                    reg_model.train(np.array([[x[0]] for x in X_train]), np.array(y_train))
                    autoencoder.train(X_train, epochs=5, batch_size=8, verbose=0)
                    print("Modelos entrenados (React conectado)")
            except Exception as e:
                print(f"Error entrenamiento ML: {e}")
        await asyncio.sleep(10)


async def keep_alive():
    
    while True:
        try:
            async with httpx.AsyncClient() as client:
                responses = await asyncio.gather(
                    client.get("https://hydrai-web-service-prod-3.onrender.com/"),
                    client.get("https://hydrapi-prod.onrender.com/"),
                    client.get("https://lstm-prod.onrender.com/"),
                    client.get("https://hydrai-net-prod.onrender.com/"),
                    return_exceptions=True
                )
            status_codes = []
            for resp in responses:
                if isinstance(resp, Exception):
                    print(f"Keep-alive error: {resp}")
                    status_codes.append(None)
                else:
                    status_codes.append(resp.status_code)
            if any(code == 200 for code in status_codes):
                print("Keep-alive exitoso")
            else:
                print(f"Keep-alive fallo: códigos recibidos {status_codes}")
        except Exception as e:
            print(f"Error keep_alive: {e}")
        await asyncio.sleep(240)


last_msg = {
    "source": "arduino",
    "tiempo": 0,
    "consumo": 0,
    "AIData": None
}

async def prepare_ai_data():
    global last_msg, last_notif_sent
    while True:
        if active_connections["react"]:
            try:
                etiqueta_actual = None
              
                msg = {
                    "source": "arduino",
                    "tiempo": hourly_time,
                    "consumo": hourly_consumption,
                    "AIData": None
                }

                if reg_model and autoencoder and len(X_train) >= 10:
                    etiqueta_actual, _ = autoencoder.predict([
                        hourly_time,
                        float(reg_model.predict([[hourly_time]])[0])
                    ])

                    pred_times = [1, 5, 10, 15, 30, 60]
                    predicciones = {}
                    for pt in pred_times:
                        future_time = hourly_time + pt
                        pred_val = float(reg_model.predict([[future_time]])[0])
                        etiqueta_pred, _ = autoencoder.predict([future_time, pred_val])
                        predicciones[str(pt)] = {
                            "prediccion": pred_val,
                            "etiqueta": etiqueta_pred
                        }

                    msg["AIData"] = {
                        "etiquetaActual": etiqueta_actual,
                        "predicciones": predicciones
                    }

               
                now = datetime.now()
                if etiqueta_actual is not None and etiqueta_actual.lower() != "ideal":
                    if (now - last_notif_sent).total_seconds() >= 60:
                        msg["notificacion"] = True
                        last_notif_sent = now
                    else:
                        msg["notificacion"] = False
                else:
                    msg["notificacion"] = False

                last_msg = msg

            except Exception as e:
                print(f"Error prepare_ai_data: {e}")

            await asyncio.sleep(1)
        else:
            await asyncio.sleep(5)



@app.websocket("/ws")
async def ws():
    global last_arduino_log, hourly_time, hourly_consumption, X_train, y_train

    token = websocket.args.get("token")
    if not token:
        await websocket.send(json.dumps({"error": "Token requerido"}))
        await websocket.close(code=1008)
        return

    payload = verify_jwt(token)
    if not payload:
        await websocket.send(json.dumps({"error": "Token inválido"}))
        await websocket.close(code=1008)
        return

    client_type = payload.get("device", payload.get("client_type", "react"))
    active_connections[client_type].add(websocket)
    print(f"Cliente {client_type} conectado")

   
    if client_type == "react" and len(active_connections["react"]) == 1:
        instantiate_models()

    try:
        async def broadcast_loop():
            while websocket in active_connections["react"]:
                try:
                  
                    await websocket.send(json.dumps(last_msg))
                except Exception as e:
                    print(f"Error broadcast: {e}")
                    break
                await asyncio.sleep(1)

        if client_type == "react":
            asyncio.create_task(broadcast_loop())

        while True:
            message = await websocket.receive()
            data = json.loads(message)

            if client_type == "arduino":
                async with lock:
                    tiempo = data.get("tiempo", 0)
                    consumo = data.get("consumo", 0)
                    hourly_time += tiempo
                    hourly_consumption += consumo
                    """
                    consumo_segundo_buffer.append({
                        "tiempo": tiempo,
                        "consumo": consumo
                    })
                    """
                    global X_train, y_train
                    X_train.append([tiempo, consumo])
                    y_train.append(consumo)
                    if len(X_train) > MAX_TRAIN_SIZE:
                        X_train = X_train[-MAX_TRAIN_SIZE:]
                        y_train = y_train[-MAX_TRAIN_SIZE:]

                now = datetime.now(mex_tz)
                if (now - last_arduino_log).total_seconds() >= 60:
                    print(f"Arduino: tiempo={tiempo}, consumo={consumo}")
                    last_arduino_log = now

                await websocket.send(json.dumps({"status": "ok"}))

            elif client_type == "react":
                if "tiempo" in data:
                    if reg_model and autoencoder and len(X_train) >= 10:
                        predicted = float(reg_model.predict([[data["tiempo"]]])[0])
                        label, mse = autoencoder.predict([data["tiempo"], predicted])
                        await websocket.send(json.dumps({
                            "status": "ok",
                            "predicted_consumption": predicted,
                            "nivelConsumo": label,
                            "reconstruction_error": mse
                        }))
                    else:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": "Modelos no entrenados"
                        }))

    except Exception as e:
        print(f"Error WebSocket {client_type}: {e}")
    finally:
        active_connections[client_type].discard(websocket)
        print(f"Cliente {client_type} desconectado")
        if client_type == "react" and not active_connections["react"]:
            destroy_models()



@app.before_serving
async def startup():
    asyncio.create_task(weather_task())
    asyncio.create_task(hourly_task())
    asyncio.create_task(ml_training_task())
    asyncio.create_task(prepare_ai_data())
    #asyncio.create_task(save_consumo_segundo_task())
    threading.Thread(target=run_daily_thread, daemon=True).start()
    asyncio.create_task(keep_alive())
    print("Tareas background iniciadas")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
