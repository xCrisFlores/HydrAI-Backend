from .controllers.user_controller import user_bp
from .controllers.sensor_controller import sensor_bp
from .controllers.consumption_controller import consumption_bp


def register_routes(app):
    app.register_blueprint(user_bp, url_prefix='/api')
    app.register_blueprint(sensor_bp, url_prefix='/api')
    app.register_blueprint(consumption_bp, url_prefix='/api')
