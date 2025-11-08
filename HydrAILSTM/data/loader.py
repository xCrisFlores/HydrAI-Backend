import os
from datetime import datetime, timedelta
from supabase import create_client
import pandas as pd
import numpy as np

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SERVICE_ROLE")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

FEATURES = [
    "tiempoActivo", "hora", "temperaturaProm",
    "sensacionProm", "humedadProm", "dia", "mes"
]
TARGET = "consumo"

def get_table_data(table, days_back=None, sensor_id=None, usuario_id=None, limit=None):

    query = supabase.table(table).select("*")
    
    if days_back:
     
        cutoff = datetime.now().replace(tzinfo=None) - timedelta(days=days_back)
        query = query.gte("fecha", cutoff.isoformat())
   
    if sensor_id:
        query = query.eq("sensor", sensor_id)
    if usuario_id:
        query = query.eq("usuario", usuario_id)
    

    query = query.order("fecha")
    
   
    if limit:
        query = query.limit(limit)
    
    response = query.execute()
    data = response.data or []
    return pd.DataFrame(data)

def preprocess_df(df, modo):
    if df.empty:
        return pd.DataFrame(columns=FEATURES + [TARGET])
    
    for col in FEATURES + [TARGET]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)
    
    if 'fecha' in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])
    
  
    if modo == "hora":
       
        return df[FEATURES + [TARGET]].copy()
    
    elif modo == "dia":
      
        df["hora"] = 0
        return df[FEATURES + [TARGET]].copy()
    
    elif modo == "semana":
       
        df["semana"] = df["fecha"].dt.isocalendar().week
        df["año"] = df["fecha"].dt.year
        
     
        groupby_cols = ["sensor", "usuario", "año", "semana"]
        available_groupby = [col for col in groupby_cols if col in df.columns]
        
        if not available_groupby:
          
            available_groupby = ["semana"]
        
        df_agg = df.groupby(available_groupby, as_index=False).agg({
            "tiempoActivo": "sum",
            "hora": "first",  
            "temperaturaProm": "mean",
            "sensacionProm": "mean",
            "humedadProm": "mean",
            "dia": "first",
            "mes": "first",
            TARGET: "sum"
        })
        
        return df_agg[FEATURES + [TARGET]]
    
    elif modo == "mes":
       
        df["año"] = df["fecha"].dt.year
        
        groupby_cols = ["sensor", "usuario", "año", "mes"]
        available_groupby = [col for col in groupby_cols if col in df.columns]
        
        if not available_groupby:
           
            available_groupby = ["mes"]
        
        df_agg = df.groupby(available_groupby, as_index=False).agg({
            "tiempoActivo": "sum",
            "hora": "first", 
            "temperaturaProm": "mean",
            "sensacionProm": "mean",
            "humedadProm": "mean",
            "dia": "first",
            "mes": "first",
            TARGET: "sum"
        })
        
        return df_agg[FEATURES + [TARGET]]
    
    else:
        
        return df[FEATURES + [TARGET]].copy()

def get_data_for_prediction(modo, sensor_id=None, usuario_id=None, look_back=7):
    
    try:
       
        table = "registrosHora" if modo == "hora" else "registrosDia"
        
       
        df = get_table_data(
            table, 
            days_back=look_back * 2,  
            sensor_id=sensor_id, 
            usuario_id=usuario_id,
            limit=look_back * 3 
        )
        
        if df.empty:
            print(f"No se encontraron datos para {modo}")
            return None
        
    
        df_processed = preprocess_df(df, modo)
        
        if len(df_processed) < look_back:
            print(f"Datos insuficientes: {len(df_processed)} < {look_back}")
            return None
        
      
        recent_data = df_processed[FEATURES].tail(look_back).values
        
        return recent_data
        
    except Exception as e:
        print(f"Error obteniendo datos para predicción: {e}")
        return None

def get_last_days_data_for_training(modo, days_back=30):

    try:
        table = "registrosHora" if modo == "hora" else "registrosDia"
        df = get_table_data(table, days_back=days_back)
        
        if df.empty:
            print(f"No se encontraron datos para {modo} en los últimos {days_back} días")
            return None, None
        
      
        df_processed = preprocess_df(df, modo)
        
        if len(df_processed) < 14: 
            print(f"Datos insuficientes para entrenamiento: {len(df_processed)} registros")
            return None, None
        
        
        data = df_processed[FEATURES].values
        target = df_processed[TARGET].values
        
        print(f"Datos obtenidos para {modo}: {len(data)} registros")
        return data, target
        
    except Exception as e:
        print(f"Error obteniendo datos para entrenamiento: {e}")
        return None, None

def get_full_training_data(modo, days_back=None):
    
    try:
        if modo in ["hora", "dia"]:
           
            table = "registrosHora" if modo == "hora" else "registrosDia"
            df = get_table_data(table, days_back=days_back)
        else:
         
            df = get_table_data("registrosDia", days_back=days_back)
        
        if df.empty:
            print(f"No se encontraron datos para {modo}")
            return None, None
        
      
        df_processed = preprocess_df(df, modo)
        
        if len(df_processed) < 14: 
            print(f"Datos insuficientes: {len(df_processed)} registros")
            return None, None
        
      
        data = df_processed[FEATURES].values
        target = df_processed[TARGET].values
        
        print(f"Datos completos para {modo}: {len(data)} registros")
        return data, target
        
    except Exception as e:
        print(f"Error obteniendo datos completos: {e}")
        return None, None

def get_data_statistics(days_back=90):
   
    stats = {}
    
    for modo, table in [("hora", "registrosHora"), ("dia", "registrosDia")]:
        try:
            df = get_table_data(table, days_back=days_back)
            
            if not df.empty:
                df['fecha'] = pd.to_datetime(df['fecha'])
                
                stats[modo] = {
                    'count': len(df),
                    'date_range': {
                        'start': df['fecha'].min().isoformat(),
                        'end': df['fecha'].max().isoformat()
                    },
                    'days_span': (df['fecha'].max() - df['fecha'].min()).days,
                    'avg_daily_records': len(df) / max(1, (df['fecha'].max() - df['fecha'].min()).days),
                    'features_stats': {
                        col: {
                            'mean': float(df[col].mean()) if col in df.columns else 0,
                            'std': float(df[col].std()) if col in df.columns else 0,
                            'min': float(df[col].min()) if col in df.columns else 0,
                            'max': float(df[col].max()) if col in df.columns else 0
                        } for col in FEATURES + [TARGET] if col in df.columns
                    }
                }
            else:
                stats[modo] = {
                    'count': 0,
                    'date_range': None,
                    'days_span': 0,
                    'avg_daily_records': 0,
                    'features_stats': {}
                }
                
        except Exception as e:
            print(f"Error obteniendo estadísticas para {modo}: {e}")
            stats[modo] = {
                'count': 0,
                'date_range': None,
                'days_span': 0,
                'avg_daily_records': 0,
                'features_stats': {},
                'error': str(e)
            }
    
    return stats

def validate_data_quality(modo, days_back=30):
    try:
        table = "registrosHora" if modo == "hora" else "registrosDia"
        df = get_table_data(table, days_back=days_back)
        
        if df.empty:
            return {
                'valid': False,
                'reason': 'No hay datos disponibles',
                'recommendations': ['Verificar conexión con base de datos', 'Revisar proceso de inserción de datos']
            }
        
        df_processed = preprocess_df(df, modo)
        
    
        total_records = len(df_processed)
        missing_values = df_processed[FEATURES + [TARGET]].isnull().sum().sum()
        zero_values = (df_processed[FEATURES + [TARGET]] == 0).sum().sum()
        
       
        issues = []
        recommendations = []
      
        min_records = 14 if modo == "dia" else 168  
        if total_records < min_records:
            issues.append(f"Datos insuficientes: {total_records} < {min_records}")
            recommendations.append(f"Recopilar al menos {min_records} registros para {modo}")
        
      
        if missing_values > 0:
            missing_pct = (missing_values / (total_records * len(FEATURES + [TARGET]))) * 100
            issues.append(f"Valores faltantes: {missing_values} ({missing_pct:.1f}%)")
            if missing_pct > 10:
                recommendations.append("Revisar proceso de recolección de datos")
        
       
        if zero_values > total_records * len(FEATURES) * 0.3:
            zero_pct = (zero_values / (total_records * len(FEATURES + [TARGET]))) * 100
            issues.append(f"Demasiados valores cero: {zero_values} ({zero_pct:.1f}%)")
            recommendations.append("Verificar sensores y proceso de medición")
        
       
        target_std = df_processed[TARGET].std()
        target_mean = df_processed[TARGET].mean()
        if target_std / target_mean < 0.1: 
            issues.append("Target con poca variabilidad")
            recommendations.append("Verificar que los datos de consumo sean correctos")
        
       
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
            df_sorted = df.sort_values('fecha')
            time_diffs = df_sorted['fecha'].diff()
            
            expected_diff = timedelta(hours=1) if modo == "hora" else timedelta(days=1)
            large_gaps = time_diffs > expected_diff * 2  
            
            if large_gaps.sum() > 0:
                issues.append(f"Gaps temporales detectados: {large_gaps.sum()}")
                recommendations.append("Revisar continuidad en la recolección de datos")
        
       
        is_valid = len(issues) == 0 or (total_records >= min_records and missing_values == 0)
        
        return {
            'valid': is_valid,
            'total_records': total_records,
            'missing_values': missing_values,
            'zero_values': zero_values,
            'issues': issues,
            'recommendations': recommendations,
            'quality_score': max(0, 100 - len(issues) * 20),  
            'target_stats': {
                'mean': float(df_processed[TARGET].mean()),
                'std': float(df_processed[TARGET].std()),
                'min': float(df_processed[TARGET].min()),
                'max': float(df_processed[TARGET].max())
            }
        }
        
    except Exception as e:
        return {
            'valid': False,
            'reason': f'Error en validación: {str(e)}',
            'recommendations': ['Revisar configuración de base de datos', 'Verificar estructura de datos']
        }

def prepare_sequence_for_prediction(raw_features, modo, look_back=7):
   
    try:
       
        df = pd.DataFrame(raw_features)
        
       
        df_processed = preprocess_df(df, modo)
        
        if len(df_processed) < look_back:
          
            if len(df_processed) > 0:
                last_row = df_processed.iloc[-1:].copy()
                while len(df_processed) < look_back:
                    df_processed = pd.concat([df_processed, last_row], ignore_index=True)
            else:
               
                zeros_data = {col: 0 for col in FEATURES}
                df_processed = pd.DataFrame([zeros_data] * look_back)
        
      
        sequence = df_processed[FEATURES].tail(look_back).values
        
        return sequence
        
    except Exception as e:
        print(f"Error preparando secuencia: {e}")
      
        return np.zeros((look_back, len(FEATURES)))

def get_recent_consumption_pattern(modo, days_back=7):
   
    try:
        table = "registrosHora" if modo == "hora" else "registrosDia"
        df = get_table_data(table, days_back=days_back)
        
        if df.empty:
            return None
        
        df_processed = preprocess_df(df, modo)
        
        if len(df_processed) == 0:
            return None
        
      
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
            df_processed['fecha'] = df['fecha']
        
      
        consumo_stats = {
            'mean': float(df_processed[TARGET].mean()),
            'std': float(df_processed[TARGET].std()),
            'min': float(df_processed[TARGET].min()),
            'max': float(df_processed[TARGET].max()),
            'trend': 'stable' 
        }
        
    
        if len(df_processed) >= 3:
            recent_values = df_processed[TARGET].tail(3).values
            if len(recent_values) >= 2:
                if recent_values[-1] > recent_values[0] * 1.1:
                    consumo_stats['trend'] = 'increasing'
                elif recent_values[-1] < recent_values[0] * 0.9:
                    consumo_stats['trend'] = 'decreasing'
        
       
        hourly_pattern = {}
        if modo == "hora" and 'hora' in df_processed.columns:
            hourly_avg = df_processed.groupby('hora')[TARGET].mean().to_dict()
            hourly_pattern = {int(k): float(v) for k, v in hourly_avg.items()}
        
     
        weekly_pattern = {}
        if 'fecha' in df_processed.columns:
            df_processed['day_of_week'] = df_processed['fecha'].dt.dayofweek
            weekly_avg = df_processed.groupby('day_of_week')[TARGET].mean().to_dict()
            weekly_pattern = {int(k): float(v) for k, v in weekly_avg.items()}
        
        return {
            'modo': modo,
            'period': f"últimos {days_back} días",
            'total_records': len(df_processed),
            'consumption_stats': consumo_stats,
            'hourly_pattern': hourly_pattern,
            'weekly_pattern': weekly_pattern,
            'last_values': df_processed[TARGET].tail(5).tolist(),
            'environmental_correlation': {
                'temperatura': float(df_processed[[TARGET, 'temperaturaProm']].corr().iloc[0, 1]) if 'temperaturaProm' in df_processed.columns else None,
                'humedad': float(df_processed[[TARGET, 'humedadProm']].corr().iloc[0, 1]) if 'humedadProm' in df_processed.columns else None
            }
        }
        
    except Exception as e:
        print(f"Error obteniendo patrón de consumo: {e}")
        return None