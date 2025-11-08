from quart import Quart, request, jsonify
from utils.auth.jwt_middleware import require_auth
from models.LSTM_HydrAI import HydrAI_LSTM
from tasks.training import start_background_training
from data.loader import get_data_for_prediction, get_last_days_data_for_training

app = Quart(__name__)

hydrai_lstm_model = HydrAI_LSTM(input_dim=7, look_back=7)

@app.before_serving
async def startup():

    start_background_training(hydrai_lstm_model)


@app.route("/api/hydrai/predict", methods=["POST"])
@require_auth()
async def predict():
    
    body = await request.get_json()
    
    sequence = body.get("sequence")
    modo = body.get("modo", "hora")
    rango = body.get("rango", "hora")
    
    if not sequence:
        return jsonify({"error": "No se proporcionó secuencia de datos"}), 400
    
    if modo not in ["hora", "dia"]:
        return jsonify({"error": "Modo inválido. Usa 'hora' o 'dia'"}), 400
    
    if rango not in ["hora", "dia", "semana", "mes"]:
        return jsonify({"error": "Rango inválido. Usa 'hora', 'dia', 'semana' o 'mes'"}), 400
    
    try:
       
        prediction = hydrai_lstm_model.predict(sequence, modo, rango)
        
        return jsonify({
            "prediction": round(prediction, 2),
        })
        
    except Exception as e:
        return jsonify({"error": f"Error en predicción: {str(e)}"}), 500

@app.route("/api/hydrai/predict-auto", methods=["POST"])
@require_auth()
async def predict_auto():
   
    body = await request.get_json()
    
    modo = body.get("modo", "hora")
    rango = body.get("rango", "dia")
    sensor_id = body.get("sensor_id")
    usuario_id = body.get("usuario_id")
    
    if modo not in ["hora", "dia"]:
        return jsonify({"error": "Modo inválido. Usa 'hora' o 'dia'"}), 400
    
    if rango not in ["hora", "dia", "semana", "mes"]:
        return jsonify({"error": "Rango inválido. Usa 'hora', 'dia', 'semana' o 'mes'"}), 400
    
    try:
     
        sequence = get_data_for_prediction(modo, sensor_id, usuario_id, look_back=7)
        
        if sequence is None or len(sequence) == 0:
            return jsonify({"error": "No hay suficientes datos históricos"}), 400
        
      
        prediction = hydrai_lstm_model.predict(sequence, modo, rango)
        
        return jsonify({
            "prediction": round(prediction, 2),
            "modo_datos": modo,
            "rango_prediccion": rango,
            "datos_historicos_usados": len(sequence),
            "unidad": "kWh" if rango in ["hora", "dia"] else f"kWh/{rango}"
        })
        
    except Exception as e:
        return jsonify({"error": f"Error en predicción automática: {str(e)}"}), 500

@app.route("/api/hydrai/finetune", methods=["POST"])
@require_auth()
async def finetune():
   
    body = await request.get_json()
    
    modo = body.get("modo", "hora")
    days_back = body.get("days_back", 30)
    epochs = body.get("epochs", 50)
    
    if modo not in ["hora", "dia"]:
        return jsonify({"error": "Modo inválido. Usa 'hora' o 'dia'"}), 400
    
    if days_back < 7 or days_back > 365:
        return jsonify({"error": "days_back debe estar entre 7 y 365"}), 400
    
    if epochs < 1 or epochs > 1000:
        return jsonify({"error": "epochs debe estar entre 1 y 1000"}), 400
    
    try:
     
        data, target = get_last_days_data_for_training(modo, days_back)
        
        if data is None or len(data) < 14:  
            return jsonify({"error": "Datos insuficientes para fine-tuning"}), 400
        
       
        history = hydrai_lstm_model.train(data, target, modo, epochs=epochs)
        
        if history is None:
            return jsonify({"error": "Error durante el entrenamiento"}), 500
        
       
        hydrai_lstm_model.save()
        
     
        final_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
        final_mae = history.history['val_mae'][-1] if 'val_mae' in history.history else None
        
        return jsonify({
            "status": "ok",
            "message": f"Modelo fine-tuned con datos de {modo}",
            "datos_entrenamiento": len(data),
            "epochs_completadas": len(history.history['loss']),
            "final_val_loss": round(final_loss, 4) if final_loss else None,
            "final_val_mae": round(final_mae, 4) if final_mae else None
        })
        
    except Exception as e:
        return jsonify({"error": f"Error en fine-tuning: {str(e)}"}), 500

@app.route("/api/hydrai/model-status", methods=["GET"])
@require_auth()
async def model_status():
    """Obtener estado del modelo"""
    try:
      
        status = {
            "modelo": "HydrAI_LSTM",
            "input_dim": hydrai_lstm_model.input_dim,
            "look_back": hydrai_lstm_model.look_back,
            "scalers_disponibles": list(hydrai_lstm_model.scalers.keys()) if hydrai_lstm_model.scalers else [],
            "modelo_inicializado": hydrai_lstm_model.model is not None
        }
        
      
        if hydrai_lstm_model.scalers:
            scaler_info = {}
            for modo, scaler in hydrai_lstm_model.scalers.items():
                scaler_info[modo] = {
                    "target_min": float(scaler['target_min']),
                    "target_max": float(scaler['target_max']),
                    "rango_target": float(scaler['target_max'] - scaler['target_min'])
                }
            status["info_scalers"] = scaler_info
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({"error": f"Error obteniendo estado: {str(e)}"}), 500

@app.route("/api/hydrai/batch-predict", methods=["POST"])
@require_auth()
async def batch_predict():
  
    body = await request.get_json()
    
    sequences = body.get("sequences", [])
    modo = body.get("modo", "hora")
    rango = body.get("rango", "hora")
    
    if not sequences or not isinstance(sequences, list):
        return jsonify({"error": "Se requiere una lista de secuencias"}), 400
    
    if len(sequences) > 100: 
        return jsonify({"error": "Máximo 100 secuencias por lote"}), 400
    
    if modo not in ["hora", "dia"]:
        return jsonify({"error": "Modo inválido. Usa 'hora' o 'dia'"}), 400
    
    if rango not in ["hora", "dia", "semana", "mes"]:
        return jsonify({"error": "Rango inválido. Usa 'hora', 'dia', 'semana' o 'mes'"}), 400
    
    try:
        predictions = []
        errors = []
        
        for i, sequence in enumerate(sequences):
            try:
                prediction = hydrai_lstm_model.predict(sequence, modo, rango)
                predictions.append({
                    "index": i,
                    "prediction": round(prediction, 2),
                    "success": True
                })
            except Exception as e:
                errors.append({
                    "index": i,
                    "error": str(e),
                    "success": False
                })
                predictions.append({
                    "index": i,
                    "prediction": None,
                    "success": False
                })
        
        return jsonify({
            "predictions": predictions,
            "total_sequences": len(sequences),
            "successful_predictions": len([p for p in predictions if p["success"]]),
            "errors": errors,
            "modo_datos": modo,
            "rango_prediccion": rango
        })
        
    except Exception as e:
        return jsonify({"error": f"Error en predicción por lotes: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)