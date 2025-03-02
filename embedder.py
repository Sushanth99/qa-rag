from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

from typing import Optional, List

class Embedder:

    def __init__(self, model_name: str, cache_folder: str = './hf_cache') -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.model = self._load_embedding_model()
        self.batch_size = 64

    
    def _load_embedding_model(self):
        """
        Loads the embedding model using the HuggingFaceEmbedding class.

        This method initializes the HuggingFaceEmbedding model with the specified
        model name, sets the trust_model_code flag to True, and specifies the cache
        folder for storing the model.

        Returns:
            HuggingFaceEmbedding: An instance of the HuggingFaceEmbedding model.
        """
        model = HuggingFaceEmbedding(model_name=self.model_name,
                                     trust_remote_code=True,
                                     cache_folder=self.cache_folder)
        return model
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using the model.

        Args:
            text (str): The input text to generate an embedding for.

        Returns:
            List[float]: A list of floats representing the generated embedding.
        """
        return self.model.get_text_embedding(text)
    
    def generate_embedding_batch(self, 
                                 batch_texts: List[str], 
                                 batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts.

        Args:
            batch_texts (List[str]): A list of text strings for which embeddings are to be generated.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        embeddings = []
        if batch_size is None:
            batch_size = self.batch_size
        
        for i in tqdm(range(0, len(batch_texts), batch_size), desc="Encoding text"):
            embeddings.extend(
                self.model.get_text_embedding_batch(batch_texts[i:i+self.batch_size])
                )
        return embeddings