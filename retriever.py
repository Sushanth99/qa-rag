
class Retriever:

    def __init__(self, vector_db, embedder):
        self.vector_db = vector_db
        self.embedder = embedder
    
    def search(self, query: str):
        query_embedding = self.embedder.generate_embedding(query)
        result = self.vector_db.search(query_embedding)
        return result