# Example modifications to remove OpenAI dependencies

import numpy as np
from sentence_transformers import SentenceTransformer
from scipy import spatial
import re


# Replace OpenAI embeddings with local sentence transformer
class OpenSourceModule:
    def __init__(self):
        # Load a local embedding model
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # Lightweight option
        # Or use: 'all-mpnet-base-v2' for better quality

    def get_local_embeddings(self, texts):
        """Replace OpenAI embeddings with local model"""
        if isinstance(texts, str):
            texts = [texts]
        return self.embedding_model.encode(texts)

    def if_two_sentence_similar_meaning_local(
        self, sentence1, sentence2, threshold=0.9
    ):
        """Replace OpenAI-based similarity with local embeddings"""
        if sentence1 == "":
            sentence1 = " "
        if sentence2 == "":
            sentence2 = " "

        embeddings = self.get_local_embeddings([sentence1, sentence2])
        score = 1 - spatial.distance.cosine(embeddings[0], embeddings[1])
        return score > threshold

    def get_top_k_similar_example_local(self, input_text, examples_df, k=4):
        """Replace OpenAI embeddings in retrieval with local model"""
        if k == 0:
            return ""

        # Get embedding for input
        input_embedding = self.get_local_embeddings(input_text)[0]

        # Compute similarities with local embeddings
        if "local_embedding" not in examples_df.columns:
            # Pre-compute embeddings for examples if not done
            examples_df["local_embedding"] = examples_df["text"].apply(
                lambda x: self.get_local_embeddings(x)[0]
            )

        examples_df["similarities"] = examples_df["local_embedding"].apply(
            lambda x: 1 - spatial.distance.cosine(x, input_embedding)
        )

        top_k_strings = examples_df.sort_values("similarities", ascending=False).head(
            k
        )["text"]

        result = ""
        for t in top_k_strings:
            if t[0] == "\n":
                t = t[1:]
            result += f"<example>\n{t}\n</example>\n\n"

        return result


# Alternative: Simple rule-based similarity for lightweight operation
def simple_similarity_check(sentence1, sentence2, threshold=0.8):
    """Lightweight alternative using token overlap"""
    # Simple token-based similarity
    tokens1 = set(re.findall(r"\w+", sentence1.lower()))
    tokens2 = set(re.findall(r"\w+", sentence2.lower()))

    if not tokens1 or not tokens2:
        return False

    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    jaccard_sim = intersection / union if union > 0 else 0
    return jaccard_sim > threshold


# Disable retrieval entirely for simplest approach
def no_retrieval_fallback():
    """Simply return empty string to disable retrieval-based examples"""
    return ""
