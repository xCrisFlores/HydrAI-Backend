import asyncio

from data.loader import get_full_training_data


def start_background_training(model):
    asyncio.create_task(_train_loop(model))

async def _train_loop(model):
    while True:
        try:
            for modo in ["hora", "dia"]:
                print(f"Entrenando en background con datos de {modo}")
                data = get_full_training_data(modo)
                if data is not None and len(data) > 0:
                    model.train(data, epochs=50)
                    model.save()
        except Exception as e:
            print(f"Error en entrenamiento background: {e}")
        await asyncio.sleep(3600) 
