"""
dispatcher.py
-------------
Dispatcher module for the Cloud2Edge AI Optimizer framework.
Detects model architecture and routes pruning requests to the appropriate engine.
"""

import logging
from typing import Dict, Tuple, Optional
from transformers import AutoConfig
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelDispatcher:
    """
    Intelligently routes model pruning requests to the appropriate engine
    based on model architecture detection.
    """
    
    # Supported model architectures
    LLM_ARCHITECTURES = {
        'bert', 'roberta', 'distilbert', 'electra',
        'gpt2', 'gpt_neo', 'gptj', 'opt',
        't5', 'bart', 'llama', 'qwen', 'mistral'
    }
    
    CNN_ARCHITECTURES = {
        'vgg', 'alexnet', 'resnet', 'efficientnet',
        'mobilenet', 'densenet', 'squeezenet'
    }
    
    def __init__(self, sire_engine=None, improvenet_engine=None):
        """
        Initialize the dispatcher with pruning engines.
        
        Args:
            sire_engine: Instance of SiRE engine for LLM pruning
            improvenet_engine: Instance of ImproveNet engine for CNN pruning
        """
        self.sire_engine = sire_engine
        self.improvenet_engine = improvenet_engine
        logger.info("ModelDispatcher initialized")
    
    def detect_model_type(self, model_name: str) -> Tuple[str, str]:
        """
        Detect whether a model is an LLM or CNN based on its name or config.
        
        Args:
            model_name: Name of the model (HuggingFace format or custom)
        
        Returns:
            Tuple of (model_type, architecture) where:
                - model_type: 'llm' or 'cnn'
                - architecture: specific architecture name
        
        Raises:
            ValueError: If model type cannot be determined
        """
        model_name_lower = model_name.lower()
        
        # Try to detect from HuggingFace config
        try:
            config = AutoConfig.from_pretrained(model_name)
            model_arch = config.model_type.lower()
            
            if model_arch in self.LLM_ARCHITECTURES:
                logger.info(f"Detected LLM: {model_arch} from HuggingFace config")
                return 'llm', model_arch
            
        except Exception as e:
            logger.debug(f"Could not load HuggingFace config: {e}")
        
        # Fallback: detect from model name
        for arch in self.LLM_ARCHITECTURES:
            if arch in model_name_lower:
                logger.info(f"Detected LLM: {arch} from model name")
                return 'llm', arch
        
        for arch in self.CNN_ARCHITECTURES:
            if arch in model_name_lower:
                logger.info(f"Detected CNN: {arch} from model name")
                return 'cnn', arch
        
        raise ValueError(
            f"Unable to detect model type for '{model_name}'. "
            f"Supported: {self.LLM_ARCHITECTURES | self.CNN_ARCHITECTURES}"
        )
    
    def dispatch(self, request_data: Dict) -> Dict:
        """
        Main dispatch method that routes requests to the appropriate engine.
        
        Args:
            request_data: Dictionary containing:
                - model_name: str, name of the model
                - task: str, task type (e.g., 'text-classification', 'image-classification')
                - pruning_ratio: float, percentage of parameters to prune (0.0 - 1.0)
                - dataset: Optional[str], dataset name for task-aware pruning
        
        Returns:
            Dictionary containing pruning results with:
                - model_type: 'llm' or 'cnn'
                - pruning_engine: 'SiRE' or 'ImproveNet'
                - delivery_method: 'json_config' or 'torchscript'
                - Additional engine-specific fields
        
        Raises:
            ValueError: If required fields are missing or engines not initialized
        """
        # Validate request
        required_fields = ['model_name', 'task']
        for field in required_fields:
            if field not in request_data:
                raise ValueError(f"Missing required field: {field}")
        
        model_name = request_data['model_name']
        task = request_data['task']
        pruning_ratio = request_data.get('pruning_ratio', 0.2)
        dataset = request_data.get('dataset', None)
        
        logger.info(f"Processing request for model: {model_name}, task: {task}")
        
        # Detect model type
        model_type, architecture = self.detect_model_type(model_name)
        
        # Route to appropriate engine
        if model_type == 'llm':
            return self._dispatch_llm(
                model_name, task, pruning_ratio, dataset, architecture
            )
        elif model_type == 'cnn':
            return self._dispatch_cnn(
                model_name, task, pruning_ratio, dataset, architecture
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _dispatch_llm(
        self, 
        model_name: str, 
        task: str, 
        pruning_ratio: float,
        dataset: Optional[str],
        architecture: str
    ) -> Dict:
        """
        Dispatch LLM pruning request to SiRE engine.
        
        Returns:
            Dictionary with JSON configuration and metadata
        """
        if self.sire_engine is None:
            raise ValueError("SiRE engine not initialized")
        
        logger.info(f"Routing to SiRE engine: {model_name}")
        
        # Call SiRE engine
        pruning_config = self.sire_engine.prune(
            model_name=model_name,
            task=task,
            pruning_ratio=pruning_ratio,
            dataset=dataset
        )
        
        # Calculate config size
        import json
        config_json = json.dumps(pruning_config)
        config_size_kb = len(config_json.encode('utf-8')) / 1024
        
        return {
            'model_type': 'llm',
            'architecture': architecture,
            'pruning_engine': 'SiRE',
            'delivery_method': 'json_config',
            'pruning_config': pruning_config,
            'file_size_kb': round(config_size_kb, 2),
            'original_model': model_name,
            'task': task,
            'pruning_ratio': pruning_ratio
        }
    
    def _dispatch_cnn(
        self, 
        model_name: str, 
        task: str, 
        pruning_ratio: float,
        dataset: Optional[str],
        architecture: str
    ) -> Dict:
        """
        Dispatch CNN pruning request to ImproveNet engine.
        
        Returns:
            Dictionary with TorchScript model path and metadata
        """
        if self.improvenet_engine is None:
            raise ValueError("ImproveNet engine not initialized")
        
        logger.info(f"Routing to ImproveNet engine: {model_name}")
        
        # Call ImproveNet engine
        result = self.improvenet_engine.prune(
            model_name=model_name,
            task=task,
            pruning_ratio=pruning_ratio,
            dataset=dataset
        )
        
        return {
            'model_type': 'cnn',
            'architecture': architecture,
            'pruning_engine': 'ImproveNet',
            'delivery_method': 'torchscript',
            'download_url': result['download_url'],
            'model_id': result['model_id'],
            'file_size_mb': result['file_size_mb'],
            'original_model': model_name,
            'task': task,
            'pruning_ratio': pruning_ratio,
            'compression_ratio': result.get('compression_ratio', None)
        }
    
    def get_supported_models(self) -> Dict:
        """
        Get list of supported model architectures.
        
        Returns:
            Dictionary with LLM and CNN architectures
        """
        return {
            'llm': sorted(list(self.LLM_ARCHITECTURES)),
            'cnn': sorted(list(self.CNN_ARCHITECTURES))
        }
    
    def validate_request(self, request_data: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate a pruning request before processing.
        
        Args:
            request_data: Request dictionary to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        required_fields = ['model_name', 'task']
        for field in required_fields:
            if field not in request_data:
                return False, f"Missing required field: {field}"
        
        # Validate pruning ratio
        pruning_ratio = request_data.get('pruning_ratio', 0.2)
        if not isinstance(pruning_ratio, (int, float)):
            return False, "pruning_ratio must be a number"
        if not 0.0 < pruning_ratio <= 1.0:
            return False, "pruning_ratio must be between 0.0 and 1.0"
        
        # Try to detect model type
        try:
            self.detect_model_type(request_data['model_name'])
        except ValueError as e:
            return False, str(e)
        
        return True, None


# Singleton instance for global access
_dispatcher_instance = None

def get_dispatcher(sire_engine=None, improvenet_engine=None) -> ModelDispatcher:
    """
    Get or create the global dispatcher instance.
    
    Args:
        sire_engine: SiRE engine instance (only needed on first call)
        improvenet_engine: ImproveNet engine instance (only needed on first call)
    
    Returns:
        ModelDispatcher instance
    """
    global _dispatcher_instance
    
    if _dispatcher_instance is None:
        _dispatcher_instance = ModelDispatcher(sire_engine, improvenet_engine)
    
    return _dispatcher_instance
