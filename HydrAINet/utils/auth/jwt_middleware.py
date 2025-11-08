from functools import wraps
from quart import request, jsonify
from utils.auth.jwt_factory import verify_jwt


def require_auth():
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            auth_header = request.headers.get("Authorization")

            if not auth_header or not auth_header.startswith("Bearer "):
                return jsonify({"error": "Token no proporcionado"}), 401

            token = auth_header.split(" ")[1]
            payload = verify_jwt(token)

            if not payload:
                return jsonify({"error": "Token inválido o expirado"}), 401

          
            request.user = payload  

            return await func(*args, **kwargs)

        return wrapper
    return decorator
