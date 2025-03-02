
## The architecture of the project is as follows:
## A class for the data and embeddings: Embed
## A class for the vector database: VectorDB
## A class for the retrieval: Retrieval
## A class for generation: RAG

from rag import RAG
from retriever import Retriever
from embedder import Embedder
from vector_database import QdrantVDB
import time

start = time.time()
emb = Embedder("nomic-ai/nomic-embed-text-v1.5", cache_folder="./hf_cache")
vdb = QdrantVDB(url="http://localhost:6333",
                collection_name="test_context",
                vector_dim=768)
retriever = Retriever(vdb, emb)
rag = RAG(retriever, 
          ollama_model_name="llama3.2", 
          ollama_request_timeout=120.0)
print(f"Setup time: {time.time()-start:.2f}s")

start = time.time()
print('Generating response...')
test_query = "What is in front of the Notre Dame Main Building?"
response = rag.query(test_query)
print(f"Total Response time: {time.time()-start:.2f}s")
print(response)
vdb.client.close()