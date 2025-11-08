from quart import Blueprint, request, jsonify
from supabase import create_client
import os
import asyncio
from dotenv import load_dotenv
from Auth.jwt_middleware import require_auth

load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
supabase = create_client(url, key)

sensor_bp = Blueprint('sensor_bp', __name__)

@sensor_bp.route('/sensores/<int:id>', methods=['GET'])
@require_auth()
async def get_by_id(id):
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("sensores").select("*").eq("id", id).execute()
        )
        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@sensor_bp.route('/sensores/usuario/<int:usuario_id>', methods=['GET'])
@require_auth()
async def get_sensores_by_usuario(usuario_id):
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("sensores").select("*").eq("usuario", usuario_id).execute()
        )
        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@sensor_bp.route('/sensores', methods=['POST'])
@require_auth()
async def create():
    try:
        data = await request.get_json()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("sensores").insert(data).execute()
        )
        return jsonify(response.data), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@sensor_bp.route('/sensores/<int:id>', methods=['PUT'])
@require_auth()
async def update(id):
    try:
        data = await request.get_json()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("sensores").update(data).eq("id", id).execute()
        )
        if response.count == 0:
            return jsonify({"error": "Sensor no encontrado"}), 404
        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@sensor_bp.route('/sensores/<int:id>', methods=['DELETE'])
@require_auth()
async def delete(id):
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("sensores").delete().eq("id", id).execute()
        )
        if response.count == 0:
            return jsonify({"error": "Sensor no encontrado"}), 404
        return jsonify({"message": "Sensor eliminado"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
