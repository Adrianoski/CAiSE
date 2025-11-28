"""
app.py
------
Flask server for Cloud2Edge AI Optimizer framework.
Main entry point for the cloud service that handles model pruning requests.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.exceptions import HTTPException
import argparse

from dispatcher import get_dispatcher
from engines.sire.sire_engine import SiREEngine
from engines.improvenet.improvenet_engine import ImproveNetEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file upload
app.config['STORAGE_PATH'] = os.getenv('STORAGE_PATH', './storage')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'cloud2edge-ai-optimizer-secret')

# Create storage directory
Path(app.config['STORAGE_PATH']).mkdir(parents=True, exist_ok=True)

# Initialize pruning engines
logger.info("Initializing pruning engines...")
sire_engine = SiREEngine()
improvenet_engine = ImproveNetEngine(storage_path=app.config['STORAGE_PATH'])

# Initialize dispatcher
dispatcher = get_dispatcher(
    sire_engine=sire_engine,
    improvenet_engine=improvenet_engine
)
logger.info("Cloud2Edge AI Optimizer initialized successfully")


# =============================
# API Endpoints
# =============================

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with service information."""
    return jsonify({
        'service': 'Cloud2Edge AI Optimizer',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'prune': '/api/prune (POST)',
            'download': '/api/download/<model_id> (GET)',
            'supported_models': '/api/models (GET)'
        },
        'documentation': 'See README.md for full API documentation'
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'engines': {
            'sire': 'active',
            'improvenet': 'active'
        }
    }), 200


@app.route('/api/models', methods=['GET'])
def get_supported_models():
    """Get list of supported model architectures."""
    try:
        supported = dispatcher.get_supported_models()
        return jsonify({
            'supported_architectures': supported,
            'llm_count': len(supported['llm']),
            'cnn_count': len(supported['cnn'])
        }), 200
    except Exception as e:
        logger.error(f"Error getting supported models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/prune', methods=['POST'])
def prune_model():
    """
    Main endpoint for model pruning requests.
    
    Expected JSON payload:
    {
        "model_name": "bert-base-uncased",
        "task": "text-classification",
        "pruning_ratio": 0.2,
        "dataset": "optional-dataset-name"
    }
    
    Returns:
        For LLMs: JSON configuration
        For CNNs: Download URL for TorchScript model
    """
    try:
        # Parse request data
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'received_content_type': request.content_type
            }), 400
        
        request_data = request.get_json()
        logger.info(f"Received pruning request: {request_data}")
        
        # Validate request
        is_valid, error_msg = dispatcher.validate_request(request_data)
        if not is_valid:
            logger.warning(f"Invalid request: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Process request through dispatcher
        result = dispatcher.dispatch(request_data)
        
        # Add server metadata
        result['server_timestamp'] = datetime.utcnow().isoformat()
        result['server_version'] = '1.0.0'
        
        logger.info(
            f"Successfully processed {result['model_type']} model "
            f"using {result['pruning_engine']} engine"
        )
        
        return jsonify(result), 200
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        logger.error(f"Error processing pruning request: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/download/<model_id>', methods=['GET'])
def download_model(model_id):
    """
    Download pruned CNN model in TorchScript format.
    
    Args:
        model_id: Unique identifier for the pruned model
    
    Returns:
        TorchScript .pt file
    """
    try:
        # Sanitize model_id to prevent directory traversal
        model_id = os.path.basename(model_id)
        
        # Construct file path
        model_path = Path(app.config['STORAGE_PATH']) / f"{model_id}.pt"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_id}")
            return jsonify({
                'error': 'Model not found',
                'model_id': model_id
            }), 404
        
        logger.info(f"Serving model: {model_id}")
        
        return send_file(
            model_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f"{model_id}.pt"
        )
    
    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {str(e)}")
        return jsonify({
            'error': 'Error downloading model',
            'message': str(e)
        }), 500


@app.route('/api/status/<model_id>', methods=['GET'])
def get_model_status(model_id):
    """
    Get status information for a pruned model.
    
    Args:
        model_id: Unique identifier for the pruned model
    
    Returns:
        Model metadata and status
    """
    try:
        model_id = os.path.basename(model_id)
        model_path = Path(app.config['STORAGE_PATH']) / f"{model_id}.pt"
        
        if not model_path.exists():
            return jsonify({
                'model_id': model_id,
                'status': 'not_found'
            }), 404
        
        # Get file info
        stat = model_path.stat()
        
        return jsonify({
            'model_id': model_id,
            'status': 'available',
            'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'download_url': f"/api/download/{model_id}"
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cleanup', methods=['POST'])
def cleanup_storage():
    """
    Clean up old pruned models from storage.
    Requires authentication (optional implementation).
    
    Expected JSON:
    {
        "older_than_hours": 24
    }
    """
    try:
        data = request.get_json() or {}
        older_than_hours = data.get('older_than_hours', 24)
        
        storage_path = Path(app.config['STORAGE_PATH'])
        deleted_count = 0
        current_time = datetime.now().timestamp()
        threshold = older_than_hours * 3600  # Convert to seconds
        
        for model_file in storage_path.glob("*.pt"):
            file_age = current_time - model_file.stat().st_mtime
            if file_age > threshold:
                model_file.unlink()
                deleted_count += 1
                logger.info(f"Deleted old model: {model_file.name}")
        
        return jsonify({
            'deleted_models': deleted_count,
            'threshold_hours': older_than_hours
        }), 200
    
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return jsonify({'error': str(e)}), 500


# =============================
# Error Handlers
# =============================

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Handle HTTP exceptions."""
    return jsonify({
        'error': e.name,
        'message': e.description
    }), e.code


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


# =============================
# Main Entry Point
# =============================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cloud2Edge AI Optimizer - Model Pruning Service'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host address to bind (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to listen on (default: 8000)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--storage',
        type=str,
        default='./storage',
        help='Path for model storage (default: ./storage)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Update config with CLI args
    app.config['STORAGE_PATH'] = args.storage
    Path(app.config['STORAGE_PATH']).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Cloud2Edge AI Optimizer on {args.host}:{args.port}")
    logger.info(f"Storage path: {app.config['STORAGE_PATH']}")
    logger.info(f"Debug mode: {args.debug}")
    
    # Run Flask app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )
