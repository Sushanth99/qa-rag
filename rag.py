from llama_index.llms.ollama import Ollama
import time

class RAG:

    def __init__(self, retriever, ollama_model_name="llama3.2:1b", ollama_request_timeout=120.0):
        self.ollama_model_name = ollama_model_name
        self.retriever = retriever
        self.ollama_request_timeout = ollama_request_timeout
        self.llm = Ollama(model=self.ollama_model_name, request_timeout=self.ollama_request_timeout)
        self.prompt_template =   """Context is given below.
                                    ---------------------
                                    {context}
                                    ---------------------
                                    
                                    Answer the following question based on the context provided above. 
                                    If you cannot answer, simply say 'I don't know.' Keep the response 
                                    short and to the point.
                                    
                                    ---------------------
                                    Query: {query}
                                    ---------------------
                                    Answer: """
    
    def generate_context(self, query):
        start_time = time.time()
        result = self.retriever.search(query)
        print(f"Retrieval time: {time.time()-start_time:.2f}")
        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context:
            context = entry["payload"]["context"]

            combined_prompt.append(context)

        return "\n\n---\n\n".join(combined_prompt)
    
    def query(self, query):
        context = self.generate_context(query=query)
        
        prompt = self.prompt_template.format(context=context,
                                                query=query)
        start_time = time.time()
        response = self.llm.complete(prompt)
        print(f"Generation time: {time.time()-start_time:.2f}")
        
        return dict(response)['text']



