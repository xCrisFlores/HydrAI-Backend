from quart import Blueprint, jsonify, request
from supabase import create_client
from datetime import datetime, timedelta
import pytz
import os
from dotenv import load_dotenv

from Auth.jwt_middleware import require_auth

load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
supabase = create_client(url, key)

mex_tz = pytz.timezone("America/Mexico_City")
consumption_bp = Blueprint('consumption_bp', __name__)


@consumption_bp.route('/registros/pruebas', methods=['GET'])
async def get_all():
    try:
        response = supabase.table("registrosPruebas").select("*").execute()
        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@consumption_bp.route('/registrosHora/', methods=['GET'])
@require_auth()
async def get_all_horas():
    """
    """
    try:

        response = supabase.table("registrosHora").select("*").execute()

        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@consumption_bp.route('/registrosDia/', methods=['GET'])
@require_auth()
async def get_all_dias():
    """
    """
    try:

        response = supabase.table("registrosDia").select("*").execute()

        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@consumption_bp.route('/registrosHora/horas', methods=['POST'])
@require_auth()
async def get_registros_rango_horas():
    """
    Devuelve registrosHora filtrados por fecha y rango de horas.
    Params (JSON body):
      - fecha: "YYYY-MM-DD"
      - horaInicio: int (0-23)
      - horaFin: int (0-23)
    """
    try:
        data = await request.get_json()
        fecha = data.get("fecha")
        hora_inicio = data.get("horaInicio")
        hora_fin = data.get("horaFin")

        if not (fecha and isinstance(hora_inicio, int) and isinstance(hora_fin, int)):
            return jsonify({"error": "Parámetros fecha, horaInicio y horaFin requeridos"}), 400

        # Ajustar a UTC (timestampz en Supabase)
        start_local = datetime.strptime(fecha, "%Y-%m-%d").replace(hour=hora_inicio)
        end_local = datetime.strptime(fecha, "%Y-%m-%d").replace(hour=hora_fin, minute=59, second=59)
        start_utc = start_local.astimezone(pytz.UTC).isoformat()
        end_utc = end_local.astimezone(pytz.UTC).isoformat()

        response = supabase.table("registrosHora").select("*") \
            .gte("fecha", start_utc) \
            .lte("fecha", end_utc) \
            .execute()

        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@consumption_bp.route('/registrosFecha/', methods=['POST'])
@require_auth()
async def get_registros_rango_fechas():
    """
    Devuelve registrosDia entre dos fechas, con análisis agregado.
    Params (JSON body):
      - fechaInicio: "YYYY-MM-DD"
      - fechaFin: "YYYY-MM-DD"
    """
    try:
        data = await request.get_json()
        fecha_inicio = data.get("fechaInicio")
        fecha_fin = data.get("fechaFin")

        if not (fecha_inicio and fecha_fin):
            return jsonify({"error": "Parámetros fechaInicio y fechaFin requeridos"}), 400

        start_local = datetime.strptime(fecha_inicio, "%Y-%m-%d").replace(hour=0)
        end_local = datetime.strptime(fecha_fin, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        start_utc = start_local.astimezone(pytz.UTC).isoformat()
        end_utc = end_local.astimezone(pytz.UTC).isoformat()

        response = supabase.table("registrosDia").select("*") \
            .gte("fecha", start_utc) \
            .lte("fecha", end_utc) \
            .execute()

        data = response.data

        if not data:
            return jsonify({"data": [], "stats": {}}), 200

        consumos = [r["consumo"] for r in data if "consumo" in r]
        tiempos = [r["tiempoActivo"] for r in data if "tiempoActivo" in r]

        stats = {
            "max": max(consumos) if consumos else None,
            "min": min(consumos) if consumos else None,
            "avgConsumo": sum(consumos) / len(consumos) if consumos else None,
            "avgTiempo": sum(tiempos) / len(tiempos) if tiempos else None,
            "total": sum(consumos) if consumos else None
        }

        return jsonify({"data": data, "stats": stats}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@consumption_bp.route('/registrosHora/', methods=['POST'])
@require_auth()
async def get_registros_por_dia():
    """
    Devuelve todos los registrosHora de un día específico, con análisis agregado.
    Params (JSON body):
      - fecha: "YYYY-MM-DD"
    """
    try:
        data = await request.get_json()
        fecha = data.get("fecha")

        if not fecha:
            return jsonify({"error": "Parámetro fecha requerido"}), 400

        # Ajustar a UTC (todo el día)
        start_local = datetime.strptime(fecha, "%Y-%m-%d").replace(hour=0)
        end_local = start_local + timedelta(days=1, seconds=-1)
        start_utc = start_local.astimezone(pytz.UTC).isoformat()
        end_utc = end_local.astimezone(pytz.UTC).isoformat()

        response = supabase.table("registrosHora").select("*") \
            .gte("fecha", start_utc) \
            .lte("fecha", end_utc) \
            .execute()

        data = response.data

        if not data:
            return jsonify({"data": [], "stats": {}}), 200

        consumos = [r["consumo"] for r in data if "consumo" in r]
        tiempos = [r["tiempoActivo"] for r in data if "tiempoActivo" in r]

        stats = {
            "max": max(consumos) if consumos else None,
            "min": min(consumos) if consumos else None,
            "avgConsumo": sum(consumos) / len(consumos) if consumos else None,
            "avgTiempo": sum(tiempos) / len(tiempos) if tiempos else None,
            "total": sum(consumos) if consumos else None
        }

        return jsonify({"data": data, "stats": stats}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@consumption_bp.route('/predict/', methods=['POST'])
@require_auth()
async def predict_regression():
    """Predicción usando la LSTM (se implementa después)"""
    return jsonify({"mensaje": "probando endpoint lstm"}), 200


@consumption_bp.route('/clasify/', methods=['POST'])
@require_auth()
async def predict_autoencoder():
    """Clasificación usando la HydrAINet (se implementa después)"""
    return jsonify({"mensaje": "probando endpoint HydrAINet"}), 200
