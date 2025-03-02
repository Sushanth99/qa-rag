from typing import List, Optional, Dict
from qdrant_client import models, QdrantClient
from tqdm import tqdm

class QdrantVDB:

    def __init__(self, url, collection_name: str, vector_dim: int = 768):
        self.url = url
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.batch_size = 64
        self.client = QdrantClient(
            url=url,
            prefer_grpc=True,
        )
    
    def create_collection(self):
        """
        Creates a new collection in the vector database if it does not already exist.

        This method checks if a collection with the specified name exists in the client.
        If the collection does not exist, it creates a new collection with the given
        configuration parameters.

        The collection is configured with the following parameters:
        - `collection_name`: The name of the collection to be created.
        - `vectors_config`: Configuration for the vectors, including size and distance metric.
        - `optimizers_config`: Configuration for the optimizers, including segment number and index threshold.

        Attributes:
            collection_name (str): The name of the collection.
            vector_dim (int): The dimensionality of the vectors in the collection.
            client (object): The client used to interact with the vector database.

        Raises:
            Exception: If there is an error creating the collection.
        """
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.DOT,
                    on_disk=True,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0,
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True),
                ),
            )

    def upsert(self, 
               vectors: List[List[float]],
               metadata: Optional[List[Dict]] = None,
               batch_size: Optional[int] = None) -> None:
        """
        Inserts vectors and their corresponding metadata into the vector database in batches.
        Args:
            vectors (List[List[float]]): A list of vectors to be inserted or updated.
            metadata (Optional[List[Dict]], optional): A list of metadata dictionaries corresponding to each vector. Defaults to None.
            batch_size (Optional[int], optional): The number of vectors to be inserted or updated in each batch. Defaults to None.
        Raises:
            AssertionError: If the length of vectors and metadata are not equal when metadata is provided.
        Returns:
            None
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if metadata:
            assert len(vectors) == len(metadata), "vectors and meta_data are of different lengths"
        
        for i in tqdm(range(0, len(vectors), batch_size), desc="Inserting into DB"):
            ## `upload_collection` is used to upload data on disk.
            if metadata:
                self.client.upload_collection(
                    collection_name=self.collection_name,
                    vectors=vectors[i:i+batch_size],
                    payload=metadata[i:i+batch_size],
                )
            else:
                self.client.upload_collection(
                    collection_name=self.collection_name,
                    vectors=vectors[i:i+batch_size],
                )
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20_000,
                )
        )

    def search(self,
               query_embedding: List[float]) -> List[List[float]]:
        """
        Searches for the nearest neighbors of the given query embedding in the vector database.
        Args:
            query_embedding (List[float]): A list of floats representing the query embedding vector.
        Returns:
            List[List[float]]: A list of lists, where each inner list represents the embedding of a nearest neighbor.
        """
        
        nbrs = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            timeout=500,
        )
        return nbrs
        

# Design questions:
#   Should we pass in the raw data or the vector data to the upsert function?
#       Ans: I believe we should pass the vector data and delegate the pre-processing
#       to another function.