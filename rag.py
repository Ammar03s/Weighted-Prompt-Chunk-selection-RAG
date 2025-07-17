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

    def __init__(self, llm_model: str = "mistral:7b", mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "chatpdf"): #mistral:7b #llama3.2:3b
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
                    "Here are the document pieces: {context}\nQuestion: {question}\nIf the document pieces are not relevant to the question, please let me know.",
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
        print(f"file path {pdf_file_path} ") 
        try:
            docs = PyPDFLoader(file_path=pdf_file_path).load()  #pdf val. happens here
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
        {"$sort": {"similarity": -1}},  #sort by similarity
        {"$limit": k},  #top k chunks
        {"$project": {"_id": 0, "index": 1, "similarity": 1}}  # Only include index and similarity
    ]
        results = list(self.collection.aggregate(pipeline))
        return results


    def generate_prompts_with_llm(self, user_prompt: str, context: str = "") -> List[str]: #context: str
        prompt_template = ChatPromptTemplate.from_template(
            "Generate exactly 5 concise and clear prompts related to the following user question. "
            "Each prompt should explore a different aspect or perspective of the question. "
            "Ensure the prompts are specific, relevant, and closely aligned with the user's intent. " #a
            "Do not include any additional text or explanations. Just provide the 5 prompts, one per line:\n"
            "User Question: {user_prompt}\n"
            "Generated Prompts (5):"
        )

        prompt_generation_chain = (prompt_template | self.model | StrOutputParser())
        generated_prompts = prompt_generation_chain.invoke({"context": context,  "user_prompt": user_prompt})
        generated_prompts_list = generated_prompts.strip().split("\n")
        generated_prompts_list = [prompt.strip() for prompt in generated_prompts_list if prompt.strip()]
        generated_prompts_list = generated_prompts_list[:5]
        
        while len(generated_prompts_list) < 5:
            generated_prompts_list.append(f"Additional prompt {len(generated_prompts_list) + 1} about {user_prompt}")
        return generated_prompts_list
    

    def get_most_relevant_chunk(self, prompt: str) -> dict:
        if not self.chunks or self.embeddings is None:
            return {"chunk_index": -1, "similarity": 0.0}
        prompt_embedding = self.embedding_model.encode(prompt, convert_to_tensor=True)
        similarities = util.cos_sim(prompt_embedding, self.embeddings).squeeze().tolist()
        max_similarity = max(similarities)
        chunk_index = similarities.index(max_similarity)
        return {"chunk_index": chunk_index, "similarity": max_similarity}


    def ask(self, query: str):
        # Step 1: Generate 5 additional prompts based on the user's query
        generated_prompts = self.generate_prompts_with_llm(query)
        weights = {"user_query": 1.05, "generated_prompts": [1.04, 1.03, 1.02, 1.01, 1.00]}

        all_results = []
        all_chunks = []
        combined_prompts = [query] + generated_prompts

        # Step 2: For each prompt (including the user's query), find the most relevant chunks
        for i, prompt in enumerate(combined_prompts):
            prompt_embedding = self.embedding_model.encode(prompt, convert_to_tensor=True)
            top_chunks = self.query_top_chunks(prompt_embedding, k=5)  # Get top 5 chunks for each prompt

            # Apply weights
            if i == 0:  # User's query
                weight = weights["user_query"]
            else:  # Generated prompts
                weight = weights["generated_prompts"][i - 1]

            for chunk in top_chunks:
                chunk["similarity"] *= weight  # Apply weight to the similarity score

            all_results.append({
                "prompt": prompt,
                "chunks": [
                    {"chunk_index": chunk["index"], "similarity": chunk["similarity"]}
                    for chunk in top_chunks
                ]
            })
            all_chunks.extend(top_chunks)

        # Step 3: Sort all chunks by weighted similarity (across all prompts)
        all_chunks_sorted = sorted(all_chunks, key=lambda x: x["similarity"], reverse=True)

        # Step 4: Deduplicate chunks by index while preserving order
        unique_chunks = []
        seen_indices = set()
        for chunk in all_chunks_sorted:
            if chunk["index"] not in seen_indices:
                unique_chunks.append(chunk)
                seen_indices.add(chunk["index"])

        # Step 5: Select the top 3 unique chunks
        top_3_chunks = unique_chunks[:3]
        top_3_indices = {chunk["index"] for chunk in top_3_chunks}

        # Step 6: Get the top 10 indices for the JSON output
        top_10_indices = {chunk["index"] for chunk in unique_chunks[:10]}

        # Step 7: Check the top 10 chunks for adjacency
        adjacent_chunks = set()

        for chunk in unique_chunks[:10]:
            chunk_index = chunk["index"]
            # Check if the chunk is adjacent to any of the top 3 chunks
            if (chunk_index + 1 in top_3_indices) or (chunk_index - 1 in top_3_indices):
                adjacent_chunks.add(chunk_index)

        # Step 8: Add adjacent chunks to the top 3 chunks
        final_chunk_indices = top_3_indices.union(adjacent_chunks)
        final_chunks = [chunk for chunk in unique_chunks if chunk["index"] in final_chunk_indices]

        # Step 9: Sort final chunks by index to maintain document order
        final_chunks_sorted = sorted(final_chunks, key=lambda x: x["index"])

        # Step 10: Use the final chunks as context to answer the user's question
        context = "\n".join([self.chunks[chunk["index"]].page_content for chunk in final_chunks_sorted])

        # Step 11: Answer the user's query using the new context
        self.chain = (RunnablePassthrough() | self.prompt | self.model | StrOutputParser())
        response = self.chain.invoke({"context": context, "question": query})

        # Step 12: json
        json_file_path = "prompty.json"
        with open(json_file_path, "w") as f:
            json.dump({
                "prompts": all_results,
                "top_3_chunks": [chunk["index"] for chunk in top_3_chunks],
                "top_10_indices": list(top_10_indices), 
                "adjacent_chunks": list(adjacent_chunks),
                "final_chunks": [chunk["index"] for chunk in final_chunks_sorted]
            }, f, indent = 4)

        return response
        

    def clear(self):
        self.embeddings = None
        self.chunks = []
        self.collection.delete_many({})