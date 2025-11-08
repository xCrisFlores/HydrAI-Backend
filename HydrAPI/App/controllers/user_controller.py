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

user_bp = Blueprint('user_bp', __name__) 

@user_bp.route('/usuarios', methods=['GET'])
@require_auth()
async def get_all():
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("usuarios").select("*").execute()
        )
        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_bp.route('/usuarios/<int:id>', methods=['GET'])
@require_auth()
async def get_by_id(id):
    try:
        loop = asyncio.get_event_loop()
        user_resp, ajustes_resp = await asyncio.gather(
            loop.run_in_executor(
                None,
                lambda: supabase.table("usuarios").select("*").eq("id", id).execute()
            ),
            loop.run_in_executor(
                None,
                lambda: supabase.table("ajustes").select("*").eq("usuario", id).execute()
            )
        )
        
        if not user_resp.data:
            return jsonify({"error": "Usuario no encontrado"}), 404
        
        user = user_resp.data[0]
        ajustes = ajustes_resp.data

        user["ajustes"] = ajustes
        return jsonify(user), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@user_bp.route('/usuarios', methods=['POST'])
@require_auth()
async def create():
    try:
        data = await request.get_json()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("usuarios").insert(data).execute()
        )
        return jsonify(response.data), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_bp.route('/login', methods=['POST'])
async def login():
    try:
        data = await request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email y contraseña requeridos"}), 400

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: supabase.table("usuarios")
                    .select("*")
                    .or_(f"email.eq.{email},username.eq.{email}")
                    .eq("password", password)
                    .execute()
        )

        if not response.data or len(response.data) == 0:
            return jsonify({"error": "Usuario no encontrado"}), 404

        user = response.data[0]
        if user["password"] != password:  
            return jsonify({"error": "Contraseña incorrecta"}), 401

        from Auth.jwt_factory import create_jwt
        token = create_jwt({
            "user_id": str(user["id"]),
            "client_type": "react"
        }, expire_in_sec=3600)

        return jsonify({
            "token": token, 
            "user": {
                "id": user["id"], 
                "username": user["username"]
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@user_bp.route('/usuarios/<int:id>', methods=['PUT'])
@require_auth()
async def update_user(id):
    try:
        data = await request.get_json()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("usuarios").update(data).eq("id", id).execute()
        )
        if response.count == 0:
            return jsonify({"error": "Usuario no encontrado"}), 404
        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_bp.route('/ajustes/<int:id>', methods=['PUT'])
@require_auth()
async def update_ajuste(id):
    try:
        data = await request.get_json()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("ajustes").update(data).eq("id", id).execute()
        )
        if response.count == 0:
            return jsonify({"error": "Ajuste no encontrado"}), 404
        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
