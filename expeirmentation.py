



#You can ignore this file, it is just for testing and experimenting purposes.





# from sentence_transformers import SentenceTransformer, util
# from langchain_core.globals import set_verbose, set_debug
# from langchain_community.chat_models import ChatOllama
# from langchain.schema.output_parser import StrOutputParser
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema.runnable import RunnablePassthrough
# from langchain_core.prompts import ChatPromptTemplate
# from pymongo import MongoClient
# import numpy as np




# set_debug(True)
# set_verbose(True)

# class ChatPDF:
#     def __init__(self, llm_model: str = "llama3.2:3b", mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "chatpdf"):
#         self.model = ChatOllama(model=llm_model)
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1024, chunk_overlap=128
#         )
#         self.prompt = ChatPromptTemplate(
#             [
#                 (
#                     "system",
#                     "You are a helpful assistant that can answer questions about the PDF document uploaded by the user.",
#                 ),
#                 (
#                     "human",
#                     "Here are the document pieces: {context}\nQuestion: {question}",
#                 ),
#             ]
#         )

#         self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
#         self.chunks = []
#         self.embeddings = None
        
#         self.client = MongoClient(mongo_uri)
#         self.db = self.client[db_name]
#         self.collection = self.db["documents"]

#     def ingest(self, pdf_file_path: str):
#         docs = PyPDFLoader(file_path=pdf_file_path).load()
#         self.chunks = self.text_splitter.split_documents(docs)

#         chunk_texts = [chunk.page_content for chunk in self.chunks]
#         self.embeddings = self.embedding_model.encode(chunk_texts, convert_to_tensor=True)

#         for i, chunk in enumerate(self.chunks):
#             self.collection.insert_one({
#                 "index": i,
#                 "content": chunk.page_content,
#                 "embedding": self.embeddings[i].tolist()  
#             })

#     def query_cosine_similarity(self, query_embedding, k=3):
#         query_embedding = query_embedding.tolist()

#         pipeline = [
#             {
#                 "$addFields": {
#                     "similarity": {
#                         "$let": {
#                             "vars": {
#                                 "dotProduct": {"$reduce": {
#                                     "input": {"$zip": {"inputs": ["$embedding", query_embedding]}},
#                                     "initialValue": 0,
#                                     "in": {"$add": [
#                                         "$$value",
#                                         {"$multiply": [{"$arrayElemAt": ["$$this", 0]}, {"$arrayElemAt": ["$$this", 1]}]}
#                                     ]}
#                                 }},
#                                 "normA": {"$sqrt": {"$reduce": {
#                                     "input": "$embedding",
#                                     "initialValue": 0,
#                                     "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
#                                 }}},
#                                 "normB": {"$sqrt": {"$reduce": {
#                                     "input": query_embedding,
#                                     "initialValue": 0,
#                                     "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
#                                 }}}
#                             },
#                             "in": {"$divide": ["$$dotProduct", {"$multiply": ["$$normA", "$$normB"]}]}
#                         }
#                     }
#                 }
#             },
#             {
#                 "$match": {
#                     "similarity": {"$gt": 0.0}
#                 }
#             },
#             {
#                 "$sort": {"similarity": -1} 
#             },
#             {
#                 "$limit": k  # İlk k sonucu al
#             }
#         ]

#         results = list(self.collection.aggregate(pipeline))
#         return results


#     def ask(self, query: str):
#         query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

#         matching_documents = self.query_cosine_similarity(query_embedding)

#         if not matching_documents:
#             return "No relevant documents found."

#         context = "\n".join([doc['content'] for doc in matching_documents])
#         print(context)
#         self.chain = (
#             RunnablePassthrough() | self.prompt | self.model | StrOutputParser()
#         )

#         response = self.chain.invoke({"context": context, "question": query})

#         return response

#     def clear(self):
#         self.embeddings = None
#         self.chunks = []
#         self.collection.delete_many({}) 


from sentence_transformers import SentenceTransformer, util
from langchain_core.globals import set_verbose, set_debug
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient

#import numpy as np
#from unstructured.partition.pdf import partition_pdf
#import pdfplumber
#import camelot
import json
from typing import List

set_debug(True)
set_verbose(True)

class ChatPDF:
    def __init__(self, llm_model: str = "llama3.2:3b", mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "chatpdf"): #mistral:7b
        self.model = ChatOllama(model = llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 128) #chunk size 600? overlap 200?
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a helpful assistant that can answer questions about the PDF document uploaded by the user.",
                ),
                (
                    "human",
                    "Here are the document pieces: {context}\nQuestion: {question}",
                ),
            ]
        )

#sentence-transformers/paraphrase-multilingual-mpnet-base-v2     
#sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2   
#sentence-transformers/distiluse-base-multilingual-v3 
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') 
        self.chunks = []
        self.embeddings = None
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["documents"]


    def ingest(self, pdf_file_path: str):
        print(f"file path {pdf_file_path} ") #p
        # if not pdf_file_path.endswith(".pdf"):
        #     raise ValueError("file must be in a pdf format") #a
        # docs = PyPDFLoader(file_path = pdf_file_path).load()
        try:
            docs = PyPDFLoader(file_path=pdf_file_path).load()  # Actual PDF validation happens here
        except Exception as e:
            raise ValueError(f"Invalid PDF: {str(e)}") #new
        
        self.chunks = self.text_splitter.split_documents(docs)
        chunk_texts = [chunk.page_content for chunk in self.chunks]
        self.embeddings = self.embedding_model.encode(chunk_texts, convert_to_tensor = True)

        for i, chunk in enumerate(self.chunks):
            self.collection.insert_one({
                "index": i,
                "content": chunk.page_content,
                "embedding": self.embeddings[i].tolist()  
            })
    #                                                 #   k=5 #decent bas still not there      #dumb
    #                                                 #   k=3 #chinese                         #decent still not there
    #                                                 #   k=7 bad                              #nah
    def query_top_chunks(self, query_embedding, k = 5):
        query_embedding = query_embedding.tolist()
        pipeline = [
        {
            "$addFields": {
                "similarity": {
                    "$let": {
                        "vars": {
                            "dotProduct": {"$reduce": {
                                "input": {"$zip": {"inputs": ["$embedding", query_embedding]}},
                                "initialValue": 0,
                                "in": {"$add": [
                                    "$$value",
                                    {"$multiply": [{"$arrayElemAt": ["$$this", 0]}, {"$arrayElemAt": ["$$this", 1]}]}
                                ]}
                            }},
                            "normA": {"$sqrt": {"$reduce": {
                                "input": "$embedding",
                                "initialValue": 0,
                                "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                            }}},
                            "normB": {"$sqrt": {"$reduce": {
                                "input": query_embedding,
                                "initialValue": 0,
                                "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                            }}}
                        },
                        "in": {"$divide": ["$$dotProduct", {"$multiply": ["$$normA", "$$normB"]}]}
                    }
                }
            }
        },
        {"$sort": {"similarity": -1}},  # Sort by similarity (descending)
        {"$limit": k},  # Limit to top k chunks
        {"$project": {"_id": 0, "index": 1, "similarity": 1}}  # Only include index and similarity
    ]
        results = list(self.collection.aggregate(pipeline))
        return results
    

    def generate_prompts_with_llm(self, user_prompt: str, context: str) -> List[str]:
        prompt_template = ChatPromptTemplate.from_template(
            "Generate exactly 5 concise and clear prompts related to the following context and user question. "
            "Do not include any additional text or explanations. Just provide the 5 prompts, one per line:\n"
            "Context: {context}\n"
            "User Question: {user_prompt}\n"
            "Generated Prompts (5):"
        )

        prompt_generation_chain = (prompt_template | self.model | StrOutputParser()) #a chain to generate prompts
        #invoke the chain to generate the prompts
        generated_prompts = prompt_generation_chain.invoke({
            "context": context,
            "user_prompt": user_prompt})

        generated_prompts_list = generated_prompts.strip().split("\n") #split prompts (assuming the LLM returns them as a block of text)
        generated_prompts_list = [prompt.strip() for prompt in generated_prompts_list if prompt.strip()] #clean the prompts, but isnt it already stripped?
        generated_prompts_list = generated_prompts_list[:5]
        while len(generated_prompts_list) < 5: #here
            generated_prompts_list.append("Default prompt (LLM did not generate enough prompts)")
        return generated_prompts_list
    
    #newwww key 
    # def calculate_average_similarity_to_chunks(self, prompt: str) -> float:
    #     if not self.chunks or self.embeddings is None: #changed it with similarities=[thingy]
    #         return 0.0 
        
    #     prompt_embedding = self.embedding_model.encode(prompt, convert_to_tensor = True)
    #     similarities = util.cos_sim(prompt_embedding, self.embeddings).squeeze().tolist() #changed it with similarities=[thingy]
    #     # similarities = []
    #     # for chunk in self.chunks: #slow?
    #     #     chunk_embedding = self.embedding_model.encode(chunk.page_content, convert_to_tensor = True)
    #     #     similarity = util.cos_sim(prompt_embedding, chunk_embedding).item()
    #     #     similarities.append(similarity)
    #     if not similarities: #changed it with similarities=[thingy]
    #         return 0.0

    #     #calc the avg similarity
    #     average_similarity = sum(similarities) / len(similarities)
    #     return average_similarity

    def get_most_relevant_chunk(self, prompt: str) -> dict:
        if not self.chunks or self.embeddings is None:
            return {"chunk_index": -1, "similarity": 0.0}
        
        prompt_embedding = self.embedding_model.encode(prompt, convert_to_tensor=True)
        similarities = util.cos_sim(prompt_embedding, self.embeddings).squeeze().tolist()
        
        max_similarity = max(similarities)
        chunk_index = similarities.index(max_similarity)
        
        return {"chunk_index": chunk_index, "similarity": max_similarity}

    def ask(self, query: str):
        # Step 1: Get top 5 chunks for the user's query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        user_top_chunks = self.query_top_chunks(query_embedding, k = 5)
        
        # Step 2: Generate context for LLM (using top 3 chunks) #from the users prompt which is what we'll modify
        context_chunks = self.query_top_chunks(query_embedding, k=3) 
        context = "\n".join([self.chunks[chunk["index"]].page_content for chunk in context_chunks])
        
        # Step 3: Generate 5 additional prompts using your existing function
        generated_prompts = self.generate_prompts_with_llm(query, context)
        
        # Step 4: Get top 5 chunks for each generated prompt
        all_results = []
        
        # Add user query results
        all_results.append({
            "prompt": query,
            "chunks": [
                {"chunk_index": chunk["index"], "similarity": chunk["similarity"]}
                for chunk in user_top_chunks
            ]
        })
        
        # Add generated prompts results
        for prompt in generated_prompts:
            prompt_embedding = self.embedding_model.encode(prompt, convert_to_tensor=True)
            top_chunks = self.query_top_chunks(prompt_embedding, k=5)
            all_results.append({
                "prompt": prompt,
                "chunks": [
                    {"chunk_index": chunk["index"], "similarity": chunk["similarity"]}
                    for chunk in top_chunks
                ]
            })
        # query_embedding = self.embedding_model.encode(query, convert_to_tensor = True)
        
        # matching_documents = self.query_top_chunks(query_embedding)
        # if not matching_documents:
        #     return "Nope, no relevant documents found, def ask"
        # context = "\n".join([doc['content'] for doc in matching_documents]) #combine the context from the matching documents
        # generated_prompts = self.generate_prompts_with_llm(query, context) #generate 5 prompts using the LLM

        # # #calc avg similarity to chunks for prompts
        # # prompt_similarities = []
        # # for i, prompt in enumerate(generated_prompts):
        # #     average_similarity = self.calculate_average_similarity_to_chunks(prompt)
        # #     print(average_similarity) #p
        # #     prompt_similarities.append({
        # #         "index": i,
        # #         "prompt": prompt,
        # #         "average_similarity": average_similarity
        # #     })
        # #     print(prompt_similarities) #p

        # # sorted_prompts = sorted(prompt_similarities, key = lambda x : x["average_similarity"], reverse = True) #sorting the prompts by avg similarity to chunks

        # # #combine the userss prompt with the generated prompts
        # # combined_prompts = [{"index": -1, "prompt": query, "average_similarity": 1.0}]  #users prompt got max simii (-1 changeable)
        # # combined_prompts.extend(sorted_prompts)

        # prompt_similarities = []
        # for prompt in generated_prompts:
        #     chunk_data = self.get_most_relevant_chunk(prompt)  # Use ONLY this
        #     prompt_similarities.append({
        #         "chunk_index": chunk_data["chunk_index"], 
        #         "prompt": prompt,
        #         "similarity": chunk_data["similarity"]
        #     })
        
        # # Sort by MAX similarity to specific chunks
        # sorted_prompts = sorted(prompt_similarities, key=lambda x: x["similarity"], reverse=True)
        
        # # Update JSON keys to match new structure
        # combined_prompts = [{"chunk_index": -1, "prompt": query, "similarity": 1.0}]
        # combined_prompts.extend(sorted_prompts)

        #json
        json_file_path = "prompt13.json" 
        with open(json_file_path, "w") as f:
            json.dump({"prompts": all_results}, f, indent = 4) #combined_prompts

        #answer user q
        self.chain = (RunnablePassthrough() | self.prompt | self.model | StrOutputParser())
        response = self.chain.invoke({"context": context, "question": query}) #query?
        return response



    def clear(self):
        self.embeddings = None
        self.chunks = []
        self.collection.delete_many({})