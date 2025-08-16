# End-To-End Medical Chat Bot

A conversational AI chatbot designed to assist users with medical queries using advanced language models and a knowledge base of medical documents.

## Features

- Natural language medical Q&A
- Contextual conversation flow
- Integration with custom medical documents (PDFs)
- User-friendly web interface
- Secure and private local deployment

## Project Structure

```
app.py                  # Main Flask app
requirements.txt        # Python dependencies
src/
   helper.py           # Helper functions
   prompt.py           # Prompt engineering
templates/
   chat.html           # Chatbot UI
static/
   style.css           # Stylesheet
data/
   Medical_book.pdf    # Medical knowledge base
model/
   llama-2-7b-chat... # Language model weights
```

## Tech Stack

- **Python**: Main programming language for backend logic
- **Flask**: Web framework for serving the chatbot application
- **HTML/CSS**: Frontend user interface
- **Llama 2 (LLM)**: Large Language Model for natural language understanding and response generation
- **Pinecone (Vector Database)**: For storing and searching document embeddings using semantic search
- **LangChain**: Framework for LLM orchestration and vector store integration
- **PyPDF or similar**: For extracting text from medical PDF documents
- **Other Python Libraries**: As specified in `requirements.txt` (e.g., transformers, llama-cpp-python, etc.)