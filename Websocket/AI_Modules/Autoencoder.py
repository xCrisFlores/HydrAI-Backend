import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import tempfile
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SERVICE_ROLE")
SUPABASE_BUCKET = "nnweights"
SUPABASE_FOLDER = "mini_autoencoder_weights"  # Carpeta dentro del bucket
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

threshold = 95
factor_ideal = 0.5
factor_normal = 1.8

class AutoencoderDetector:
    def __init__(
        self,
        input_dim=2,
        encoding_dim=4,
        threshold_percentile=threshold,
        initial_model_name="initial_mini_ae.weights.h5",
        current_model_name="mini_ae.weights.h5",
    ):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.initial_model_name = initial_model_name
        self.current_model_name = current_model_name
        self.model_name = None
        self.model = self._build_model()
        self.threshold = None

    
        if not self._download_weights(self.current_model_name):
            if not self._download_weights(self.initial_model_name):
                print("No se encontraron pesos en Supabase. Modelo vacío.")
                return
        self.load()

    def _build_model(self):
        input_layer = keras.Input(shape=(self.input_dim,))
        encoded = layers.Dense(self.encoding_dim, activation="relu")(input_layer)
        decoded = layers.Dense(self.input_dim, activation="linear")(encoded)
        autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer="adam", loss="mse")
        return autoencoder

    def train(self, X_train, epochs=50, batch_size=16, verbose=0):
        X_train = np.array(X_train)
        self.model.fit(
            X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=verbose
        )
        reconstructions = self.model.predict(X_train, verbose=0)
        mse = np.mean(np.square(X_train - reconstructions), axis=1)
        self.threshold = np.percentile(mse, self.threshold_percentile)
        print(f"Entrenado. Umbral establecido: {self.threshold:.6f}")

    def predict(self, X):
        X = np.array(X).reshape(1, -1)
        reconstruction = self.model.predict(X, verbose=0)
        mse = np.mean(np.square(X - reconstruction))
        if mse < self.threshold * factor_ideal:
                return "ideal", mse
        elif mse < self.threshold * factor_normal:
            return "normal", mse
        else:
            return "alto", mse


    def save(self):
       
        self.model_name = self.current_model_name
        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
            self.model.save_weights(tmp.name)
            tmp_path = tmp.name

        remote_path = f"{SUPABASE_FOLDER}/{self.current_model_name}"

        try:
          
            files = supabase.storage.from_(SUPABASE_BUCKET).list(SUPABASE_FOLDER)
            if any(f["name"] == self.current_model_name for f in files):
                supabase.storage.from_(SUPABASE_BUCKET).remove([remote_path])
                print(f"Archivo remoto {remote_path} eliminado antes de subir")

            with open(tmp_path, "rb") as f:
                response = supabase.storage.from_(SUPABASE_BUCKET).upload(
                    remote_path, f, {"content-type": "application/octet-stream"}
                )

            if response.get("error"):
                print(f"Error subiendo pesos: {response['error']}")
            else:
                print(f"Pesos subidos como: {remote_path}")

        except Exception as e:
            print(f"Error al subir archivo: {e}")
        finally:
            os.remove(tmp_path)

    def load(self):
       
        if not self.model_name or not os.path.exists(self.model_name):
            print(f"Archivo {self.model_name} no existe para cargar")
            return
        self.model.load_weights(self.model_name)
        print(f"Pesos cargados localmente desde: {self.model_name}")

    def _download_weights(self, file_name):
       
        remote_path = f"{SUPABASE_FOLDER}/{file_name}"
        try:
          
            result = supabase.storage.from_(SUPABASE_BUCKET).list(SUPABASE_FOLDER)
            if isinstance(result, dict) and "error" in result:
                print(f"Error al listar archivos: {result['error']}")
                return False

            if not any(f["name"] == file_name for f in result):
                print(f"{file_name} no existe en Supabase")
                return False

           
            response = supabase.storage.from_(SUPABASE_BUCKET).download(remote_path)

            if isinstance(response, dict) and "error" in response:
                print(f"Error descargando {file_name}: {response['error']}")
                return False

        
            with open(file_name, "wb") as f:
                f.write(response)

            self.model_name = file_name
            print(f"Pesos descargados: {file_name}")
            return True
        except Exception as e:
            print(f"Error descargando {file_name}: {e}")
            return False

