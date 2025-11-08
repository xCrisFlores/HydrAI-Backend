from dotenv import load_dotenv
from quart import Quart
from quart_cors import cors
from App.routes import register_routes
from Auth.jwt_factory import create_jwt

load_dotenv()

app = Quart(__name__)
app = cors(app, allow_origin="*")


register_routes(app)

@app.before_serving
async def startup():

    print("Servidor iniciando...")
    print(create_jwt(payload={
            "sensor_id": 1,
            "device": "arduino"
            }, forEsp=True))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
