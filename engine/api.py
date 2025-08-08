from flask import Blueprint, jsonify, request, render_template
from .modernbert_processing import ModernBERTProcessing
import os
import logging

api = Blueprint('api', __name__)
LOG = logging.getLogger(__name__)
modernbert_proc = ModernBERTProcessing()

@api.route('/', methods=['GET'])
def buy_me_a_coffee():
    ascii_art = '''
 ░█▄█░█▀▄░█▀▀░█▀▄░▀█▀░░░░░█▀▀░█▄█░█▀▄░█▀▀░█▀▄░█▀▄░▀█▀░█▀█░█▀▀░█▀▀
 ░█░█░█▀▄░█▀▀░█▀▄░░█░░▄▄▄░█▀▀░█░█░█▀▄░█▀▀░█░█░█░█░░█░░█░█░█░█░▀▀█
 ░▀░▀░▀▀░░▀▀▀░▀░▀░░▀░░░░░░▀▀▀░▀░▀░▀▀░░▀▀▀░▀▀░░▀▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀    
          "author":"Piotr Romanowski"
          "version": "2.1"
    '''
    return render_template('index.html', ascii_art=ascii_art)

@api.route('/ver', methods=['GET'])
def version():
    log_event()
    return jsonify({"version": "2.1", "author":"Piotr Romanowski","compiled":"11-06-2025 17:53","served_by":os.uname()[2], "container":os.path.exists('/.dockerenv')})

@api.route('/health', methods=['GET'])
def health():
    log_event()
    return jsonify({"status": "ok"})

@api.route('/v1/embeddings', methods=['POST'])
def create_embedding():
    j_input = request.get_json()
    embedding = modernbert_proc.embedding(text_list=j_input['input'])
    log_event()
    return jsonify(embedding)

def log_event():
    hostname = os.uname()[1]
    LOG.info('served from {}'.format(os.uname()[2]))
