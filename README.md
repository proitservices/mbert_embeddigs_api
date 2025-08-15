# mBERT Embeddings API

A lightweight API for generating 1024-token embeddings using `answerdotai/ModernBERT-base`, optimized for low-power systems like the N100 cluster.

## Setup

### Prerequisites
- Docker
- Python 3.11 (due to `torch.compile` compatibility)
- Model cache: ~500MB for `answerdotai/ModernBERT-base`

### Model Caching
1. Cache the model:
   ```bash
   python3.11 -m venv /path/to/venv
   source /path/to/venv/bin/activate
   pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
   pip install transformers==4.48.0
   export HF_HOME=/mnt/raid/models/mbert_embeddings #location of preference
   python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('answerdotai/ModernBERT-base'); AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')"
