from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernBERTProcessing:
    def __init__(self, model_name='answerdotai/ModernBERT-base'):
        """
        Initialize the ModernBERT-base model for generating embeddings.
        Args:
            model_name (str): Name of the Hugging Face model.
        """
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self.device = torch.device('cpu')  # Use CPU for N100
        self.model.to(self.device)
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

    def embedding(self, text_list):
        """
        Generate embeddings for a list of texts, one embedding per text.
        Args:
            text_list (list): List of strings to embed.
        Returns:
            dict: OpenAI-style response with embeddings.
        """
        logger.info(f"Processing {len(text_list)} texts")
        start_time = time.time()

        embeddings_list = []
        batch_size = 8  # Fixed batch size for efficiency
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_texts)} texts")
            batch_start = time.time()

            # Tokenize and encode batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=8192
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = sum_embeddings / sum_mask
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings_list.extend(embedding.cpu().numpy())

            logger.info(f"Batch processed in {time.time() - batch_start:.2f} seconds")

        logger.info(f"Total embedding time: {time.time() - start_time:.2f} seconds")
        return self.transform_embedding_to_dict(embeddings_list, text_list)

    def transform_embedding_to_dict(self, embedding_list, text_list, model_name="text-embedding-modernbert-base-001"):
        """
        Transform embeddings into OpenAI-compatible dictionary format.
        Args:
            embedding_list (list): List of embedding vectors.
            text_list (list): List of input texts.
            model_name (str): Name for the model in the response.
        Returns:
            dict: Formatted response.
        """
        prompt_tokens = sum(len(text) for text in text_list)  # Characters
        total_tokens = len(embedding_list) * 768  # 768 dimensions per embedding
        logger.info(f"Generated {len(embedding_list)} embeddings, total_tokens={total_tokens}")

        transformed_data = {
            "data": [
                {
                    "embedding": embedding.tolist(),
                    "index": index,
                    "object": "embedding"
                }
                for index, embedding in enumerate(embedding_list)
            ],
            "model": model_name,
            "object": "list",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
        }
        return transformed_data