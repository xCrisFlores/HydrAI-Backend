import asyncio
import time
from datetime import datetime
from data.loader import get_full_training_data

def start_background_training(model):
    asyncio.create_task(_train_loop(model))

async def _train_loop(model):
  
    iteration = 0
    
    while True:
        try:
            iteration += 1
         
            for modo in ["hora", "dia"]:
                try:
                    print(f"Obteniendo datos para modo: {modo}")
                    
                  
                    data, target = get_full_training_data(modo, days_back=30)
                    
                    if data is not None and len(data) > 14: 
                        print(f"Datos obtenidos para {modo}: {len(data)} registros")
                        
                        
                        epochs = 30 if iteration == 1 else 20  
                        
                        history = model.train(
                            data, target, modo, 
                            epochs=epochs, 
                            validation_split=0.15  
                        )
                        
                        if history:
                           
                            final_loss = history.history.get('val_loss', [0])[-1]
                            final_mae = history.history.get('val_mae', [0])[-1]
                            
                            print(f"Entrenamiento {modo} completado:")
                            print(f"Épocas: {epochs}")
                            print(f"Val Loss: {final_loss:.4f}")
                            print(f"Val MAE: {final_mae:.4f}")
                        else:
                            print(f"Entrenamiento {modo} falló")
                    else:
                        print(f"Datos insuficientes para {modo}: {len(data) if data is not None else 0} registros")
                
                except Exception as e:
                    print(f"Error entrenando modo {modo}: {e}")
                
              
                await asyncio.sleep(5)
            
           
            try:
                model.save()
                print("Modelo guardado en Supabase")
            except Exception as e:
                print(f"Error guardando modelo: {e}")
            
            
            if iteration == 1:
               
                wait_time = 1800 
            elif iteration <= 5:

                wait_time = 3600  
            else:

                wait_time = 7200  
            
            print(f"Próximo entrenamiento en {wait_time//60} minutos")
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            print(f"Error crítico en loop de entrenamiento: {e}")
          
            await asyncio.sleep(600) 
async def manual_training_session(model, modo="mixed", days_back=60, epochs=100):
   
    modos_a_entrenar = ["hora", "dia"] if modo == "mixed" else [modo]
    results = {}
    
    for current_modo in modos_a_entrenar:
        try:
            print(f"\nEntrenando modo: {current_modo}")
            
          
            data, target = get_full_training_data(current_modo, days_back=days_back)
            
            if data is None or len(data) < 21: 
                print(f"Datos insuficientes para {current_modo}")
                continue
            
            print(f"Datos obtenidos: {len(data)} registros")
            
           
            start_time = time.time()
            history = model.train(
                data, target, current_modo,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2
            )
            training_time = time.time() - start_time
            
            if history:
              
                final_loss = history.history.get('val_loss', [0])[-1]
                final_mae = history.history.get('val_mae', [0])[-1]
                total_epochs = len(history.history.get('loss', []))
                
                results[current_modo] = {
                    'final_val_loss': final_loss,
                    'final_val_mae': final_mae,
                    'total_epochs': total_epochs,
                    'training_time': training_time,
                    'data_points': len(data)
                }
                
                print(f"Entrenamiento {current_modo} completado:")
                print(f"Épocas completadas: {total_epochs}")
                print(f"Tiempo: {training_time:.1f}s")
                print(f"Val Loss final: {final_loss:.4f}")
                print(f"Val MAE final: {final_mae:.4f}")
            else:
                print(f"Entrenamiento {current_modo} falló")
                
        except Exception as e:
            print(f"Error en entrenamiento manual {current_modo}: {e}")
    
    if results:
        try:
            model.save()
            print("Modelo guardado exitosamente")
        except Exception as e:
            print(f"Error guardando modelo: {e}")
    
   
    if results:
        print(f"\nResumen de entrenamiento manual:")
        for modo_res, metrics in results.items():
            print(f"   {modo_res}:")
            print(f"Datos: {metrics['data_points']} registros")
            print(f"Épocas: {metrics['total_epochs']}")
            print(f"Tiempo: {metrics['training_time']:.1f}s")
            print(f"Val Loss: {metrics['final_val_loss']:.4f}")
            print(f"Val MAE: {metrics['final_val_mae']:.4f}")
    
    return results

def schedule_intensive_training(model):
    asyncio.create_task(_intensive_training_loop(model))
    print("Entrenamiento intensivo programado")

async def _intensive_training_loop(model):
  
    while True:
        try:
            await asyncio.sleep(604800)
            
            print("Iniciando entrenamiento intensivo semanal")
            
         
            results = await manual_training_session(
                model,
                modo="mixed",
                days_back=90,  
                epochs=200    
            )
            
            if results:
                print("Entrenamiento intensivo semanal completado")
            else:
                print("Entrenamiento intensivo semanal sin resultados")
                
        except Exception as e:
            print(f"Error en entrenamiento intensivo: {e}")
          
            await asyncio.sleep(86400)

def get_training_recommendations(data_stats):
   
    recommendations = {
        'look_back': 7,     
        'epochs': 100,      
        'batch_size': 32,   
        'days_back': 60     
    }
    
   
    total_points = sum(stats.get('count', 0) for stats in data_stats.values())
    
    if total_points > 1000:
        recommendations['look_back'] = 14  
    elif total_points > 500:
        recommendations['look_back'] = 10  
    else:
        recommendations['look_back'] = 7   
    
  
    if total_points > 2000:
        recommendations['epochs'] = 200
    elif total_points > 1000:
        recommendations['epochs'] = 150
    elif total_points < 200:
        recommendations['epochs'] = 50
    
  
    if total_points > 1000:
        recommendations['batch_size'] = 64
    elif total_points < 200:
        recommendations['batch_size'] = 16
    
   
    max_days_span = max(stats.get('days_span', 30) for stats in data_stats.values())
    recommendations['days_back'] = min(max_days_span, 90)
    
    return recommendations