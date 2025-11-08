from quart import Quart, request, jsonify
from utils.auth.jwt_middleware import require_auth
from models.HydrAI_net import HydrAI
from tasks.training import start_background_training
from data.loader import get_data_for_range, get_last_30_days_data

app = Quart(__name__)
hydr_ai_model = HydrAI(input_dim=8) 

@app.before_serving
async def startup():
    start_background_training(hydr_ai_model)

@app.route("/api/hydrai/classify", methods=["POST"])
@require_auth()
async def classify():
    body = await request.get_json()
    features = body.get("features")
    rango = body.get("rango", "hora") 

    if not features:
        return jsonify({"error": "No se proporcionaron features"}), 400

    try:
        scaled = get_data_for_range(rango, features)
        cluster = hydr_ai_model.classify(scaled)
        return jsonify({"cluster": cluster})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/hydrai/finetune", methods=["POST"])
@require_auth()
async def finetune():
    body = await request.get_json()
    modo = body.get("modo", "hora")  

    if modo not in ["hora", "dia", "semana", "mes"]:
        return jsonify({"error": "Modo inválido. Usa 'hora', 'dia', 'semana' o 'mes'"}), 400

    try:
        data = get_last_30_days_data(modo)
        hydr_ai_model.train(data, epochs=10)
        hydr_ai_model.save()
        return jsonify({"status": "ok", "msg": f"Modelo actualizado con datos de {modo}."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import asyncio
    asyncio.run(app.run_task(host="0.0.0.0", port=8000))
