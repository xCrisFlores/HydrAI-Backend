import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from supabase import create_client

# Optimizaciones de memoria para TensorFlow
tf.config.experimental.enable_memory_growth = True
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SERVICE_ROLE")
SUPABASE_BUCKET = "nnweights"
SUPABASE_FOLDER = "lstm_weights"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class HydrAI_LSTM:
    def __init__(self, input_dim, look_back=3):  
        self.input_dim = int(input_dim)
        self.look_back = look_back
        self.model_filename = "HydrAI_LSTM_light.weights.h5"
        self.scaler_filename = "HydrAI_LSTM_scaler_light.npz"
        
        self.data_min = None
        self.data_max = None
        self.target_min = None
        self.target_max = None
        
        self.model = None
        self.build_model()
        
        if not self._download_weights():
            print("No se encontraron pesos en Supabase. Modelo vacío.")
        else:
            self.load()

    def build_model(self):
        
        tf.keras.backend.clear_session()
        inputs = Input(shape=(self.look_back, self.input_dim), name="lstm_input")
        x = LSTM(16, return_sequences=True, dropout=0.1, name="lstm_1")(inputs)
        x = LSTM(8, return_sequences=False, dropout=0.1, name="lstm_2")(x)
        x = Dense(4, activation='relu', name="dense_1")(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1, activation='linear', name="prediction")(x)
        self.model = Model(inputs=inputs, outputs=outputs, name="HydrAI_LSTM_Light")
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
        print(f"Parámetros del modelo: {self.model.count_params():,}")
        return self.model

    def create_sequences(self, data, target, look_back):
      
        data_len = len(data) - look_back
        if data_len <= 0:
            return np.array([]), np.array([])
    
        X = np.zeros((data_len, look_back, data.shape[1]), dtype=np.float32)
        Y = np.zeros((data_len,), dtype=np.float32)
        
        for i in range(data_len):
            X[i] = data[i:i+look_back]
            Y[i] = target[i+look_back]
        
        return X, Y

    def normalize_data(self, data, target, fit=True):
        if fit or self.data_min is None:
            self.data_min = np.min(data, axis=0)
            self.data_max = np.max(data, axis=0)
            self.target_min = np.min(target)
            self.target_max = np.max(target)
    
        data_range = self.data_max - self.data_min
        data_range = np.where(data_range == 0, 1, data_range)
        target_range = self.target_max - self.target_min
        target_range = 1 if target_range == 0 else target_range
        
      
        data_norm = ((data - self.data_min) / data_range).astype(np.float32)
        target_norm = ((target - self.target_min) / target_range).astype(np.float32)
        
        return data_norm, target_norm

    def denormalize_prediction(self, prediction):
        
        if self.target_min is None or self.target_max is None:
            return prediction
        
        target_range = self.target_max - self.target_min
        target_range = 1 if target_range == 0 else target_range
        
        return prediction * target_range + self.target_min

    def train(self, data, target, modo, epochs=100, batch_size=16, validation_split=0.2):
        data = np.array(data, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        
        if len(data) < self.look_back + 1:
            print(f"Datos insuficientes para modo {modo}: {len(data)} < {self.look_back + 1}")
            return None
        
        data_norm, target_norm = self.normalize_data(data, target, fit=True)
        
        X_seq, Y_seq = self.create_sequences(data_norm, target_norm, self.look_back)
        
        if len(X_seq) == 0:
            print(f"No se pudieron crear secuencias para modo {modo}")
            return None
        
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        Y_train, Y_test = Y_seq[:split_idx], Y_seq[split_idx:]
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=20, 
                restore_best_weights=True,
                verbose=0
            )
        ]
       
        history = self.model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, Y_test),
            callbacks=callbacks,
            verbose=1
        )
        
       
        val_loss = self.model.evaluate(X_test, Y_test, verbose=0)
        return history

    def predict(self, x, modo, rango_objetivo="hora"):
        x = np.array(x, dtype=np.float32)

        if len(x.shape) == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        elif len(x.shape) == 1:
            x = x.reshape(1, 1, x.shape[0])
        
  
        if x.shape[1] != self.look_back:
            if x.shape[1] > self.look_back:
                x = x[:, -self.look_back:, :]
            else:
        
                last_obs = x[:, -1:, :]
                padding = np.repeat(last_obs, self.look_back - x.shape[1], axis=1)
                x = np.concatenate([x, padding], axis=1)
        
     
        if self.data_min is not None:
            data_range = self.data_max - self.data_min
            data_range = np.where(data_range == 0, 1, data_range)
            x = (x - self.data_min) / data_range
        
       
        pred_norm = self.model.predict(x, verbose=0, batch_size=1)
        pred = self.denormalize_prediction(pred_norm)
        
        scaling_factors = {
            "hora": 1.0,
            "dia": 24.0,
            "semana": 168.0,
            "mes": 720.0
        }
        
        if modo == "hora" and rango_objetivo != "hora":
            pred = pred * scaling_factors[rango_objetivo]
        elif modo == "dia" and rango_objetivo == "hora":
            pred = pred / scaling_factors["dia"]
        elif modo == "dia" and rango_objetivo in ["semana", "mes"]:
            pred = pred * (scaling_factors[rango_objetivo] / scaling_factors["dia"])
        
        return float(pred[0][0])

    def save(self):
       
        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
            self.model.save_weights(tmp.name)
            tmp_path = tmp.name
        
     
        scaler_data = {
            'data_min': self.data_min,
            'data_max': self.data_max,
            'target_min': self.target_min,
            'target_max': self.target_max
        }
        
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_scaler:
            np.savez(tmp_scaler.name, **scaler_data)
            tmp_scaler_path = tmp_scaler.name
        
        remote_model_path = f"{SUPABASE_FOLDER}/{self.model_filename}"
        remote_scaler_path = f"{SUPABASE_FOLDER}/{self.scaler_filename}"
        
        try:
          
            with open(tmp_path, "rb") as f:
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    remote_model_path, f, 
                    {"content-type": "application/octet-stream", "x-upsert": "true"}
                )
            
            with open(tmp_scaler_path, "rb") as f:
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    remote_scaler_path, f, 
                    {"content-type": "application/octet-stream", "x-upsert": "true"}
                )
            
            print(f"Modelo guardado en Supabase")
            
        except Exception as e:
            print(f"Error subiendo archivos: {e}")
        finally:
            os.remove(tmp_path)
            os.remove(tmp_scaler_path)

    def load(self):
        """Cargar modelo optimizado"""
        try:
            if os.path.exists(self.model_filename):
                self.model.load_weights(self.model_filename)
                print(f"Pesos cargados desde {self.model_filename}")
            
            if os.path.exists(self.scaler_filename):
                scaler_data = np.load(self.scaler_filename, allow_pickle=True)
                
                if 'data_min' in scaler_data:
                    self.data_min = scaler_data['data_min']
                    self.data_max = scaler_data['data_max']
                    self.target_min = float(scaler_data['target_min'])
                    self.target_max = float(scaler_data['target_max'])
                    print(f"📦 Scalers cargados desde {self.scaler_filename}")
                
        except Exception as e:
            print(f"Error cargando modelo: {e}")

    def _download_weights(self):
      
        remote_model_path = f"{SUPABASE_FOLDER}/{self.model_filename}"
        remote_scaler_path = f"{SUPABASE_FOLDER}/{self.scaler_filename}"
        
        try:
            files = supabase.storage.from_(SUPABASE_BUCKET).list(SUPABASE_FOLDER)
            if "error" in files:
                return False
            
            file_names = [f["name"] for f in files]
            success = True
            
            if self.model_filename in file_names:
                response = supabase.storage.from_(SUPABASE_BUCKET).download(remote_model_path)
                with open(self.model_filename, "wb") as f:
                    f.write(response)
                print(f"Pesos descargados: {self.model_filename}")
            else:
                success = False
            
            if self.scaler_filename in file_names:
                response = supabase.storage.from_(SUPABASE_BUCKET).download(remote_scaler_path)
                with open(self.scaler_filename, "wb") as f:
                    f.write(response)
                print(f"Scalers descargados: {self.scaler_filename}")
            
            return success
            
        except Exception as e:
            print(f"Error descargando desde Supabase: {e}")
            return False
