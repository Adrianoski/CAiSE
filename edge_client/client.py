"""
client.py
---------
Edge client for requesting model pruning from Cloud2Edge AI Optimizer service.
Provides a simple interface for edge devices to optimize models via the cloud.
"""

import requests
import json
import logging
from typing import Dict, Optional, Union
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Cloud2EdgeClient:
    
    def __init__(self, server_url: str, timeout: int = 300):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"Initialized Cloud2Edge client for server: {self.server_url}")
    
    def health_check(self) -> bool:
        try:
            response = requests.get(
                f"{self.server_url}/api/health",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Service health: {data['status']}")
                return True
            return False
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def get_supported_models(self) -> Optional[Dict]:
        try:
            response = requests.get(
                f"{self.server_url}/api/models",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            logger.info(
                f"Supported models - LLMs: {data['llm_count']}, "
                f"CNNs: {data['cnn_count']}"
            )
            return data
        except Exception as e:
            logger.error(f"Failed to get supported models: {str(e)}")
            return None
    
    def request_pruning(
        self,
        model_name: str,
        task: str,
        pruning_ratio: float = 0.2,
        dataset: Optional[str] = None
    ) -> Optional[Dict]:
        payload = {
            'model_name': model_name,
            'task': task,
            'pruning_ratio': pruning_ratio
        }
        
        if dataset:
            payload['dataset'] = dataset
        
        logger.info(f"Requesting pruning for {model_name} (ratio: {pruning_ratio})")
        
        try:
            response = requests.post(
                f"{self.server_url}/api/prune",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            logger.info(
                f"Pruning completed - Type: {result['model_type']}, "
                f"Engine: {result['pruning_engine']}, "
                f"Delivery: {result['delivery_method']}"
            )
            
            if result['delivery_method'] == 'json_config':
                logger.info(f"Config size: {result['file_size_kb']} KB")
            elif result['delivery_method'] == 'torchscript':
                logger.info(f"Model size: {result['file_size_mb']} MB")
            
            return result
        
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout} seconds")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Failed to request pruning: {str(e)}")
            return None
    
    def download_model(
        self,
        model_id: str,
        save_path: Union[str, Path]
    ) -> bool:
        try:
            logger.info(f"Downloading model {model_id}...")
            
            response = requests.get(
                f"{self.server_url}/api/download/{model_id}",
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.debug(f"Download progress: {progress:.1f}%")
            
            logger.info(f"Model downloaded successfully to {save_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            return False
    
    def get_model_status(self, model_id: str) -> Optional[Dict]:
        try:
            response = requests.get(
                f"{self.server_url}/api/status/{model_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model status: {str(e)}")
            return None
    
    def prune_llm(
        self,
        model_name: str,
        task: str,
        pruning_ratio: float = 0.2,
        dataset: Optional[str] = None,
        save_config_path: Optional[Union[str, Path]] = None
    ) -> Optional[Dict]:
        result = self.request_pruning(model_name, task, pruning_ratio, dataset)
        
        if result and result.get('delivery_method') == 'json_config':
            config = result.get('pruning_config')
            
            if save_config_path and config:
                save_path = Path(save_config_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Pruning config saved to {save_path}")
            
            return config
        
        return None
    
    def prune_cnn(
        self,
        model_name: str,
        task: str,
        pruning_ratio: float = 0.25,
        dataset: Optional[str] = None,
        save_model_path: Optional[Union[str, Path]] = None
    ) -> bool:
        result = self.request_pruning(model_name, task, pruning_ratio, dataset)
        
        if result and result.get('delivery_method') == 'torchscript':
            model_id = result.get('model_id')
            
            if save_model_path and model_id:
                return self.download_model(model_id, save_model_path)
            
            return True
        
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cloud2Edge AI Optimizer - Edge Client'
    )
    parser.add_argument(
        '--server',
        type=str,
        required=True,
        help='Cloud service URL (e.g., http://localhost:8000)'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name to prune'
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='Task type (e.g., text-classification, image-classification)'
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.2,
        help='Pruning ratio (default: 0.2)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name for task-aware pruning'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for pruned model/config'
    )
    parser.add_argument(
        '--health',
        action='store_true',
        help='Only check service health'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List supported models'
    )
    
    args = parser.parse_args()
    
    client = Cloud2EdgeClient(args.server)
    
    if args.health:
        is_healthy = client.health_check()
        exit(0 if is_healthy else 1)
    
    if args.list_models:
        models = client.get_supported_models()
        if models:
            print("\n=== Supported Models ===")
            print(f"\nLLMs ({models['llm_count']}):")
            for arch in models['supported_architectures']['llm']:
                print(f"  - {arch}")
            print(f"\nCNNs ({models['cnn_count']}):")
            for arch in models['supported_architectures']['cnn']:
                print(f"  - {arch}")
        exit(0)
    
    print(f"\n=== Requesting pruning for {args.model} ===")
    result = client.request_pruning(
        model_name=args.model,
        task=args.task,
        pruning_ratio=args.ratio,
        dataset=args.dataset
    )
    
    if not result:
        print("Pruning request failed")
        exit(1)
    
    print(f"Pruning successful!")
    print(f"   Model type: {result['model_type']}")
    print(f"   Engine: {result['pruning_engine']}")
    print(f"   Delivery: {result['delivery_method']}")
    
    if args.output:
        if result['delivery_method'] == 'json_config':
            with open(args.output, 'w') as f:
                json.dump(result['pruning_config'], f, indent=2)
            print(f"   Config saved to: {args.output}")
        
        elif result['delivery_method'] == 'torchscript':
            model_id = result['model_id']
            success = client.download_model(model_id, args.output)
            if success:
                print(f"   Model saved to: {args.output}")
            else:
                print("   Failed to download model")
                exit(1)
    
    print("\n=== Summary ===")
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
