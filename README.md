# A cloud-edge framework for optimising AI models as-a-service

## About

This framework is a lightweight and modular system that enables neural network pruning via a cloud-edge architecture. Edge devices send model names and tasks to the server, which intelligently dispatches requests to the appropriate pruning engine based on model architecture. The system returns either a JSON pruning configuration (for LLMs) or a ready-to-deploy TorchScript model (for CNNs), all through a simple asynchronous API.

Designed for computing continuum environments, the system supports both **Transformer-based LLMs** (BERT, GPT, RoBERTa) and **CNNs** (ResNet, VGG, EfficientNet) from Hugging Face Hub.

## Architecture Overview
```
Edge Device → Cloud Service → Dispatcher → SiRE (LLM) → JSON config (~KB)
                                        → ImproveNet (CNN) → TorchScript model (~MB)
```

- **Dispatcher**: Detects model architecture and routes to appropriate engine
- **SiRE**: Analyzes LLMs and returns lightweight JSON with pruning indices
- **ImproveNet**: Prunes CNNs in the cloud and exports ready-to-use TorchScript models

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cloud-edge-pruning-framework.git
cd cloud-edge-pruning-framework
```

### 2. Install dependencies
```bash
# For cloud service
pip install -r requirements-cloud.txt

# For edge devices (lightweight)
pip install -r requirements-edge.txt
```

## Project Structure
```
.
├── cloud_service/             # Core cloud logic and APIs
│   ├── app.py                 # Flask server - launch this to start the service
│   ├── dispatcher.py          # Model type detection and routing logic
│   ├── engines/
│   │   ├── sire/              # LLM pruning engine (returns JSON config)
│   │   │   ├── analyzer.py
│   │   │   └── config_generator.py
│   │   └── improvenet/        # CNN pruning engine (returns TorchScript)
│   │       ├── pruner.py
│   │       └── torchscript_exporter.py
│   └── storage/               # Temporary storage for pruned CNN models
│
├── edge_client/               # Edge device utilities
│   ├── client.py              # Script to request model pruning
│   ├── llm_config_applier.py  # Apply JSON config to LLMs
│   └── model_loader.py        # Load TorchScript CNNs
│
├── examples/                  # Usage examples
│   ├── edge_llm_example.py
│   └── edge_cnn_example.py
│
├── requirements-cloud.txt     # Cloud service dependencies
├── requirements-edge.txt      # Edge device dependencies
└── README.md
```

## How to Run the Cloud Service

Navigate to the cloud_service directory and launch the server:
```bash
cd cloud_service
python app.py --port 8000
```

The server will be ready to receive pruning requests via REST API.

## Edge Client Usage

### For LLMs (Receive JSON Configuration)
```python
import requests
from transformers import AutoModel
from edge_client.llm_config_applier import apply_pruning_config

# Request pruning configuration
payload = {
    "model_name": "bert-base-uncased",
    "task": "text-classification",
    "pruning_ratio": 0.2
}

response = requests.post(
    "http://cloud-service:8000/api/prune",
    json=payload
)

config = response.json()
print(f"Delivery: {config['delivery_method']}")  # json_config
print(f"Size: {config['file_size_kb']} KB")

# Download original model and apply config
model = AutoModel.from_pretrained("bert-base-uncased")
pruned_model = apply_pruning_config(model, config['pruning_config'])
```

### For CNNs (Receive Pruned Model)
```python
import requests
import torch

# Request pruned model
payload = {
    "model_name": "VGG16",
    "task": "image-classification",
    "pruning_ratio": 0.25
}

response = requests.post(
    "http://cloud-service:8000/api/prune",
    json=payload
)

config = response.json()
print(f"Delivery: {config['delivery_method']}")  # torchscript
print(f"Size: {config['file_size_mb']} MB")

# Download and load pruned model
model_response = requests.get(config['download_url'])
with open("pruned_model.pt", "wb") as f:
    f.write(model_response.content)

model = torch.jit.load("pruned_model.pt")
model.eval()
```

## API Endpoints

### POST `/api/prune`

Main endpoint for requesting model pruning.

**Request:**
```json
{
  "model_name": "string",
  "task": "string",
  "pruning_ratio": 0.2
}
```

**Response (LLM):**
```json
{
  "model_type": "llm",
  "pruning_engine": "SiRE",
  "delivery_method": "json_config",
  "pruning_config": {...}
}
```

**Response (CNN):**
```json
{
  "model_type": "cnn",
  "pruning_engine": "ImproveNet",
  "delivery_method": "torchscript",
  "download_url": "http://.../model.pt"
}
```

### GET `/api/download/{model_id}`

Download pruned CNN model (TorchScript format).

### GET `/api/health`

Health check endpoint.

## Supported Models

### LLMs (SiRE Engine)
- BERT, RoBERTa, DistilBERT, ELECTRA
- GPT-2, GPT-Neo
- T5, BART

**Delivery**: JSON configuration (~5-20 KB)

### CNNs (ImproveNet Engine)
- VGG16, VGG19
- AlexNet

**Delivery**: TorchScript model (~20-100 MB)

## About the Pruning Engines

### SiRE (Structured Importance-based Reduction Engine)

The SiRE engine analyzes Transformer-based models in the cloud and generates a lightweight JSON configuration containing indices of attention heads and feed-forward neurons to prune. Edge devices download the original model from Hugging Face and apply the configuration locally.

### ImproveNet

The core ImproveNet algorithm performs complete CNN pruning in the cloud and exports optimized TorchScript models ready for deployment. The algorithm is proprietary and not included in this repository, but the backend is modular and designed to support custom compression logic.

## Data Transfer

| Model Type | Delivery Format | Typical Size | Edge Processing |
|-----------|----------------|--------------|-----------------|
| LLM | JSON config | 5-20 KB | Apply config to original model |
| CNN | TorchScript .pt | 20-100 MB | Load directly, no processing |

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
