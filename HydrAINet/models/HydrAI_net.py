import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SERVICE_ROLE")
SUPABASE_BUCKET = "nnweights"
SUPABASE_FOLDER = "autoencoder_weights"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class HydrAI:
    def __init__(self, input_dim, latent_dim=1, n_clusters=3):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.model_filename = "HydrAI.weights.h5"

        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.kmeans = None
        self.latent_reps = None
        self.clusters = None

        self.model = self._build_model()

        if not self._download_weights():
            print("No se encontraron pesos en Supabase. Modelo vacío.")
        else:
            self.load()

    def _build_model(self):
        input_layer = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(16, activation="relu")(input_layer)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(8, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        latent = layers.Dense(self.latent_dim, activation="linear", name="latent_space")(x)
        
        x = layers.Dense(8, activation="relu")(latent)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(16, activation="relu")(x)
        output_layer = layers.Dense(self.input_dim, activation="sigmoid")(x)

        self.encoder = keras.Model(inputs=input_layer, outputs=latent, name="encoder")

        decoder_input = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(8, activation="relu")(decoder_input)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(16, activation="relu")(x)
        decoder_output = layers.Dense(self.input_dim, activation="sigmoid")(x)
        self.decoder = keras.Model(inputs=decoder_input, outputs=decoder_output, name="decoder")

        autoencoder = keras.Model(inputs=input_layer, outputs=output_layer, name="autoencoder")
        autoencoder.compile(optimizer="adam", loss="mse")

        self.autoencoder = autoencoder
        return autoencoder

    def train(self, X_train, epochs=300, batch_size=64, verbose=1):
        X_train = np.array(X_train)
        self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=verbose,
            validation_split=0.1
        )
        self.latent_reps = self.encoder.predict(X_train)
        self._cluster_latent()
        print(f"Entrenado y clusterizado: {np.unique(self.clusters, return_counts=True)}")

    def _cluster_latent(self):
        if self.latent_reps is None:
            raise ValueError("No hay representaciones latentes para clusterizar")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
        self.clusters = self.kmeans.fit_predict(self.latent_reps)

    def classify(self, x):
        x = np.array(x).reshape(1, -1)
        latent = self.encoder.predict(x)
        if self.kmeans is None:
            raise ValueError("Modelo KMeans no entrenado")
        return int(self.kmeans.predict(latent)[0])

    def save(self):
        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
            self.autoencoder.save_weights(tmp.name)
            tmp_path = tmp.name

        remote_path = f"{SUPABASE_FOLDER}/{self.model_filename}"
        try:
            with open(tmp_path, "rb") as f:
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    remote_path, f, {"content-type": "application/octet-stream", "x-upsert": "true"}
                )
            print(f"Pesos guardados en: {remote_path}")
        except Exception as e:
            print(f"Error subiendo archivo: {e}")
        finally:
            os.remove(tmp_path)

    def load(self):
        if not os.path.exists(self.model_filename):
            print(f"Archivo local {self.model_filename} no existe para cargar")
            return
        self.autoencoder.load_weights(self.model_filename)
        print(f"Pesos cargados localmente desde {self.model_filename}")

    def _download_weights(self):
        remote_path = f"{SUPABASE_FOLDER}/{self.model_filename}"
        try:
            files = supabase.storage.from_(SUPABASE_BUCKET).list(SUPABASE_FOLDER)
            if "error" in files:
                print(f"Error listando archivos: {files['error']}")
                return False
            found = any(f["name"] == self.model_filename for f in files)
            if not found:
                print(f"Archivo {self.model_filename} no encontrado en Supabase")
                return False
            response = supabase.storage.from_(SUPABASE_BUCKET).download(remote_path)
            with open(self.model_filename, "wb") as f:
                f.write(response)
            print(f"Pesos descargados: {self.model_filename}")
            return True
        except Exception as e:
            print(f"Error descargando pesos: {e}")
            return False
