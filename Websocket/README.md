# HydrAI_WS
Este es proyecto es un Web Socket que integra distintos clientes, como los sensores y las aplicaciones o clientes, de forma que la interaccion entre estos sea bidireccional, el origen de los datos es el del ESP8266, un circuito con una tarjeta de red que envia los datos de un sensor (actualmente simulados) al socket, mientras que este, se encarga de procesar los datos recibidos, ademas de entrenar y utilizar una regresion en tiempo real, para poder enviar estos datos, ademas de la prediccion, a todos los clientes que esten conectados por medio de un broadcast.
## ¿Como iniciar el proyecto?
Para iniciar el proyecto o servicio del socket, puedes iniciar clonando este repositorio usando el siguiente comando git:
```
git clone https://github.com/xCrisFlores/HydrAI_WS.git
```
O en su defecto simplemente descarga el proyecto desde esta pagina, una vez que tengas el proyecto dirigete a el en tu explorador de archivos y abre una terminal, o navega a el desde una terminal si conoces la ruta, ahora necesitas activar el entorno virtual del proyecto con el siguiente comando:
```
venv\Scripts\activate
```
Y posteriormente instalar sus dependencias:
```
pip install -r requirements.txt
```
Una vez que hayas instalado las dependencias, dirigete al archivo app.py y remplaza con tu IP la palabra IP, una vez que hayas cofigurado la ruta de la API puedes iniciar el servicio con el siguiente comando:

```
flask run
```
Lo que iniciara el servicio del socket, es necesario que ademas consultes la documentacion e instrucciones de los demas proyectos e iniciarlos, ya que este servicio podria no funcioanr sin los demas, a continuacion se enlistan los demas proyectos necesarios:
## Otros proyectos necesarios
* [HydrAPI](https://github.com/xCrisFlores/HydrAPI) (API para interactuar con la base de datos)
* [HydrAI](https://github.com/xCrisFlores/HydrAI) (Aplicacion movil para visualizar los datos generados por el sensor)

## Puntos importantes
Si vas a colaborar en este repositorio, el proyecto suele generar carpetas de cache, antes de subir tus cambios elimina estas carpetas y regresa la linea de codigo de la url de la IP como estaba antes "http://IP:5050/api/consumption" esto para evitar conflictos entre versiones, o siempre puedes crear una nueva branch para subir tus cambios
