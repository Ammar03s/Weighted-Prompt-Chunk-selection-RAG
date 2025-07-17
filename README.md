# ChatPDF - Optimized Home made RAG

**Developed optimized RAG system with custom intelligent algorithm for enhanced prompt engineering and intelligent chunk selection**

## Overview

This project implements a new Retrieval Augmented Generation (RAG) system i created that goes beyond traditional similarity-based document retrieval. The system features a sophisticated multi-prompt generation strategy combined with intelligent chunk selection algorithms to provide more accurate and contextually relevant responses to the pdf..

## Technical Architecture

### Enhanced Prompt Engineering

The system employs a **weighted multi-prompt strategy** that generates multiple query variations to capture different semantic aspects of the user's question:

1. **Primary Query Processing**: User input receives the highest weight (1.05)
2. **LLM-Generated Prompts**: Five additional prompts are generated using the base LLM, each exploring different perspectives of the original query
3. **Weighted Scoring**: Generated prompts receive decreasing weights to prioritize relevancy

### Intelligent Chunk Selection Algorithm

The chunk selection process implements a **multi-stage ranking and adjacency detection system**:

#### Stage 1: Multi-Prompt Retrieval
- For each prompt (user + 5 generated), retrieve top 5 most similar chunks
- Apply weight multipliers to similarity scores
- Aggregate all results across prompts

#### Stage 2: Deduplication and Ranking
- Sort all chunks by weighted similarity scores
- Remove duplicates while preserving highest-scored instances
- Select top 3 unique chunks as primary context

#### Stage 3: Adjacency Enhancement
- Identify chunks adjacent to top 3 chunks (±1 index positions)
- Include adjacent chunks to maintain document continuity
- Combine primary and adjacent chunks for final context

#### Stage 4: Context Assembly
- Sort final chunk set by document order
- Concatenate content to form coherent context
- Generate response using enhanced context

### Key Features

- **MongoDB Integration**: Efficient vector storage and cosine similarity search
- **Sentence Transformers**: Multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Ollama LLM**: Local Mistral 7B model integration 
- **Streamlit Interface**: Clean, responsive web interface
- **Comprehensive Logging**: JSON output tracking prompt generation and chunk selection

## Project Structure

```
simple-rag/
├── rag.py              # Core RAG implementation with advanced algorithms
├── app.py              # Streamlit web interface
├── prompty.json        # Output file showing prompt generation and chunk selection process
└── experimentation/    # Testing and development files
```

### File Descriptions

- **`rag.py`**: Contains the `ChatPDF` class with enhanced prompt engineering and chunk selection algorithms
- **`app.py`**: Streamlit web application providing user interface for PDF upload and chat functionality  
- **`prompty.json`**: Generated output file demonstrating the prompt creation process, chunk rankings, and selection logic for analysis and debugging
- **Experimentation files**: For development and testing purposes (included it cuz why not)

## Installation

### Prerequisites
- Python
- MongoDB
- Ollama with Mistral 7B model (or whatever model u prefer)

### Setup Instructions

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and configure Ollama**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull mistral:7b #or whatever model u prefer
   ```

4. **Start MongoDB**
   ```bash
   # Start MongoDB service
   mongod
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload PDF**: Use the file uploader to select a PDF document
2. **Wait for Processing**: The system will extract text, generate embeddings, and store in MongoDB
3. **Ask Questions**: Enter questions about the document content
4. **Review Process**: Check `prompty.json` to see prompt generation and chunk selection details



## License

MIT Licens

## Contributing

This project focuses on R&D of new Intelligent RAG techniques. 
Contributions to algorithm improvements and performance optimizations are of course welcome.
