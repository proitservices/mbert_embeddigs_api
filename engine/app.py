from flask import Flask
from engine.api import api
import logging
from datetime import datetime

def configure_logging():
    logfile = './logs/modernbert_embedding_api_{}.log'.format(datetime.now().strftime("%Y%m%d"))
    # register root logging
    logging.basicConfig(filename=logfile, level=logging.INFO)

def create_app(settings_overrides=None):
    app = Flask(__name__)
    configure_settings(app, settings_overrides)
    configure_logging()
    configure_blueprints(app)
    return app

def configure_settings(app, settings_override):
    app.config.update({
        'INFO': True,
        'DEBUG': False,
        'TESTING': False,
    })
    if settings_override:
        app.config.update(settings_override)

def configure_blueprints(app):
    app.register_blueprint(api)