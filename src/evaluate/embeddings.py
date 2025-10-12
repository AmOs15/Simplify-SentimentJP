"""
Sentence embedding computation for semantic similarity evaluation.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import EvaluatorConfig


class EmbeddingComputer:
    """Computes and caches sentence embeddings for similarity evaluation."""

    def __init__(self, config: EvaluatorConfig):
        """Initialize the embedding computer.

        Args:
            config: Evaluator configuration
        """
        self.config = config
        self.device = self._get_device()
        self.model = self._load_model()

    def _get_device(self) -> str:
        """Determine the device to use.

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model.

        Returns:
            Loaded SentenceTransformer model
        """
        model = SentenceTransformer(self.config.embedding_model_name)
        model.to(self.device)
        return model

    def _get_cache_path(self, texts: List[str], prefix: str = "") -> Path:
        """Get the cache file path for a list of texts.

        Args:
            texts: List of texts to embed
            prefix: Prefix for the cache file name

        Returns:
            Path to the cache file
        """
        # Create a hash of the texts to use as cache key
        text_hash = hashlib.md5(
            json.dumps(texts, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

        model_name_safe = self.config.embedding_model_name.replace("/", "_")
        cache_filename = f"{prefix}_{model_name_safe}_{text_hash}.npy"
        return self.config.cache_dir / cache_filename

    def compute_embeddings(
        self,
        texts: List[str],
        prefix: str = "",
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            prefix: Prefix for the cache file name
            use_cache: Whether to use cached embeddings if available
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        cache_path = self._get_cache_path(texts, prefix)

        # Try to load from cache
        if use_cache and cache_path.exists():
            embeddings = np.load(cache_path)
            if show_progress:
                print(f"Loaded embeddings from cache: {cache_path}")
            return embeddings

        # Compute embeddings
        if show_progress:
            print(f"Computing embeddings for {len(texts)} texts...")

        # Process in batches
        embeddings_list = []
        for i in tqdm(
            range(0, len(texts), self.config.batch_size),
            disable=not show_progress,
            desc="Computing embeddings",
        ):
            batch_texts = texts[i : i + self.config.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device,
            )
            embeddings_list.append(batch_embeddings)

        embeddings = np.vstack(embeddings_list)

        # Convert precision if needed
        if self.config.precision == "float16":
            embeddings = embeddings.astype(np.float16)
        else:
            embeddings = embeddings.astype(np.float32)

        # Save to cache
        if use_cache:
            np.save(cache_path, embeddings)
            if show_progress:
                print(f"Saved embeddings to cache: {cache_path}")

        return embeddings

    def compute_cosine_similarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (n_texts, embedding_dim)
            embeddings2: Second set of embeddings (n_texts, embedding_dim)

        Returns:
            Array of cosine similarities with shape (n_texts,)
        """
        # Normalize embeddings
        embeddings1_norm = embeddings1 / np.linalg.norm(
            embeddings1, axis=1, keepdims=True
        )
        embeddings2_norm = embeddings2 / np.linalg.norm(
            embeddings2, axis=1, keepdims=True
        )

        # Compute cosine similarity (element-wise dot product)
        cosine_sim = np.sum(embeddings1_norm * embeddings2_norm, axis=1)

        return cosine_sim

    def compute_similarity_for_pairs(
        self,
        original_texts: List[str],
        simplified_texts: List[str],
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute cosine similarity for pairs of original and simplified texts.

        Args:
            original_texts: List of original texts
            simplified_texts: List of simplified texts
            use_cache: Whether to use cached embeddings
            show_progress: Whether to show progress bar

        Returns:
            Array of cosine similarities with shape (n_pairs,)
        """
        if len(original_texts) != len(simplified_texts):
            raise ValueError(
                f"Number of original texts ({len(original_texts)}) must match "
                f"number of simplified texts ({len(simplified_texts)})"
            )

        # Compute embeddings for original and simplified texts
        original_embeddings = self.compute_embeddings(
            original_texts, prefix="original", use_cache=use_cache, show_progress=show_progress
        )
        simplified_embeddings = self.compute_embeddings(
            simplified_texts, prefix="simplified", use_cache=use_cache, show_progress=show_progress
        )

        # Compute cosine similarity
        cosine_similarities = self.compute_cosine_similarity(
            original_embeddings, simplified_embeddings
        )

        return cosine_similarities
