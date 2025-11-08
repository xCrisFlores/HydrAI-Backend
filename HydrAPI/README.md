# HydrAI_WS
Este es proyecto es una API que permite la interaccion de los datos y el codigo por medio de un ORM (SQLAlchemy) a una base de datos SQL Server. Esta API permite gestionar distintas peticiones para recuperar informacion o crear nuevos registros en la base de datos.
## ¿Como iniciar el proyecto?
Antes de poder iniciar la API es necesario que cuentes con SQLServer o alguna otra distribucion de SQL, (El proyeto esta configurado para utilizar SQL Server, con la finalidad de migrar a un servicio en la nube), puedes descargar SQL Server del siguiente enalce:
Antes de poner en marcha este script es necesario que cuentes con las siguientes tecnologias instaladas:
* [SQL Server](https://www.microsoft.com/es-mx/sql-server/sql-server-downloads) (Opcional o alguna otra distribucion de SQL)
* [SQL Server Management Studio](https://learn.microsoft.com/en-us/ssms/download-sql-server-management-studio-ssms)


Una vez que hays instalado SQL Server o cualquier otra distribucion es necesario que hagas una base de datos, con el nombre de "HydrAI_DB" o el nombre que desees, solo toma nota de la cadena de conexion para esta base datos.

Para iniciar la API, necesitas clonar este repositorio usando el siguiente comando git:
```
git clone https://github.com/xCrisFlores/HydrAPI.git
```
O en su defecto simplemente descarga el proyecto desde esta pagina, una vez que tengas el proyecto dirigete a el en tu explorador de archivos y abre una terminal, o navega a el desde una terminal si conoces la ruta, ahora necesitas activar el entorno virtual del proyecto con el siguiente comando:
```
venv\Scripts\activate
```
Y posteriormente instalar sus dependencias:
```
pip install -r requirements.txt
```
Una vez que hayas instalado las dependencias, dirigete al archivo config.py dentro de la carpeta App y en el parametro SQLALCHEMY_DATABASE_URI ingresa tu cadena de conexion, una vez que hayas cofigurado esta cadena puedes iniciar el servicio con el siguiente comando:

```
python run.py
```
Lo que iniciara el servicio de la API, es necesario que ademas consultes la documentacion e instrucciones de los demas proyectos e iniciarlos. Por si mismo, este proyecto no depende de los demas, pero para hacer pruebas completas necesitas de los siguiente proyectos:
## Otros proyectos necesarios
* [HydrAI_WS](https://github.com/xCrisFlores/HydrAI_WS) (Socket para distribucion de datos en tiempo real)
* [HydrAI](https://github.com/xCrisFlores/HydrAI) (Aplicacion movil para visualizar los datos generados por el sensor)

## Puntos importantes
* Si vas a colaborar en este repositorio, el proyecto suele generar carpetas de cache, antes de subir tus cambios elimina estas carpetas y regresa la linea de codigo de la cadena de conexioncomo estaba antes, esto para evitar conflictos entre versiones, o siempre puedes crear una nueva branch para subir tus cambios.
* Antes de iniciar las pruebas es necesario que hagas un registro en la tabla de usuarios, y otro en la de sensores.
* Este proyecto incluye un archivo llamado script_arduino.txt este archivo es el archivo de configuracion para el ESP8266 deberas modificar algunos parametros como el id del usuario y sensor que acabaste de generar, para que este pueda compilar correctamente en el circuito y ademas funcione correctamente entre todo el flujo de servicios.