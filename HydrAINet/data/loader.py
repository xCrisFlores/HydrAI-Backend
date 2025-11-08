import os
from datetime import datetime, timedelta
from supabase import create_client
import pandas as pd
import numpy as np

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SERVICE_ROLE")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

FEATURES = [
    "consumo", "tiempoActivo", "hora", "temperaturaProm",
    "sensacionProm", "humedadProm", "dia", "mes"
]

feature_minmax = {}

def normalize(df, modo):
    global feature_minmax
    if modo not in feature_minmax:
        min_vals = df.min()
        max_vals = df.max()
        feature_minmax[modo] = (min_vals, max_vals)
    else:
        min_vals, max_vals = feature_minmax[modo]

    return (df - min_vals) / (max_vals - min_vals + 1e-8)

def get_table_data(table, days_back=None):
    query = supabase.table(table).select("*")
    if days_back:
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        query = query.gte("fecha", cutoff.isoformat())
    response = query.execute()
    data = response.data or []
    return pd.DataFrame(data)

def preprocess_df(df, modo):
    if df.empty:
        return pd.DataFrame(columns=FEATURES)

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    if modo in ["dia", "semana", "mes"]:
        df["hora"] = 0

    if modo == "semana":

        df["fecha"] = pd.to_datetime(df["fecha"])
        df["semana"] = df["fecha"].dt.isocalendar().week
     
        df_agg = df.groupby(["sensor", "usuario", "mes", "semana"], as_index=False).agg({
            "tiempoActivo": "sum",
            "hora": "first",  # es 0 ya
            "temperaturaProm": "mean",
            "sensacionProm": "mean",
            "humedadProm": "mean",
            "dia": "first",
        })
        df_agg["mes"] = df_agg["mes"].fillna(0)
        return df_agg[FEATURES]

    elif modo == "mes":
        df["fecha"] = pd.to_datetime(df["fecha"])
       
        df_agg = df.groupby(["sensor", "usuario", "mes"], as_index=False).agg({
            "tiempoActivo": "sum",
            "hora": "first",  
            "temperaturaProm": "mean",
            "sensacionProm": "mean",
            "humedadProm": "mean",
            "dia": "first",
        })
        df_agg["mes"] = df_agg["mes"].fillna(0)
        return df_agg[FEATURES]

    else:
     
        return df[FEATURES]

def get_data_for_range(rango, features):
    df = pd.DataFrame([features])
    df_processed = preprocess_df(df, rango)
    df_norm = normalize(df_processed, rango)
    return df_norm.to_numpy()

def get_last_30_days_data(modo):
    if modo == "hora":
        df = get_table_data("registrosHora", days_back=30)
    elif modo == "dia":
        df = get_table_data("registrosDia", days_back=30)
    elif modo == "semana" or modo == "mes":
      
        df = get_table_data("registrosDia", days_back=30)
    else:
      
        df_horas = get_table_data("registrosHora", days_back=30)
        df_dias = get_table_data("registrosDia", days_back=30)
        df = pd.concat([df_horas, df_dias], ignore_index=True)

    df_processed = preprocess_df(df, modo)
    df_norm = normalize(df_processed, modo)
    return df_norm.to_numpy()

def get_full_training_data(modo):
    if modo == "hora":
        df = get_table_data("registrosHora")
    elif modo == "dia":
        df = get_table_data("registrosDia")
    elif modo == "semana" or modo == "mes":
        df = get_table_data("registrosDia")
    else:
        df_horas = get_table_data("registrosHora")
        df_dias = get_table_data("registrosDia")
        df = pd.concat([df_horas, df_dias], ignore_index=True)

    df_processed = preprocess_df(df, modo)
    df_norm = normalize(df_processed, modo)
    return df_norm.to_numpy()
