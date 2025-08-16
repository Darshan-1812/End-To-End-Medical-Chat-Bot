from functools import wraps
def patch_pinecone_query(index):
    orig_query = index.query
    @wraps(orig_query)
    def new_query(*args, **kwargs):
        if args:
            # Assume first arg is vector, second is top_k, third is namespace
            if len(args) == 1:
                kwargs['vector'] = args[0]
            elif len(args) == 2:
                kwargs['vector'] = args[0]
                kwargs['top_k'] = args[1]
            elif len(args) == 3:
                kwargs['vector'] = args[0]
                kwargs['top_k'] = args[1]
                kwargs['namespace'] = args[2]
            args = ()
        return orig_query(*args, **kwargs)
    index.query = new_query
    return index
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone as PineconeClient
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = download_hugging_face_embeddings()


# Initializing Pinecone using the new client syntax
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"  # Updated to match actual Pinecone index name

# Create Pinecone index and patch query method for compatibility
index = pc.Index(index_name)
index = patch_pinecone_query(index)

# Loading the index with LangChain's Pinecone vectorstore wrapper
from langchain.vectorstores import Pinecone as LangchainPinecone
docsearch = LangchainPinecone(index, embeddings.embed_query, "text")


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')




@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"Received message: {msg}")
    try:
        input = msg
        print(f"Querying QA chain with: {input}")
        # Step 1: Try Pinecone retrieval
        try:
            print("[DEBUG] Calling docsearch.as_retriever().get_relevant_documents...")
            docs = docsearch.as_retriever(search_kwargs={'k': 2}).get_relevant_documents(input)
            print(f"[DEBUG] Retrieved docs: {docs}")
        except Exception as e:
            print(f"[ERROR] Pinecone retrieval failed: {e}")
            return "Sorry, there was a problem retrieving information from the database."

        # Step 2: Try LLM QA chain
        try:
            print("[DEBUG] Calling QA chain...")
            result = qa({"query": input})
            print(f"[DEBUG] Raw QA result: {result}")
            answer = result.get("result", "No answer returned.")
            print(f"[DEBUG] Final answer: {answer}")
            return str(answer)
        except Exception as e:
            print(f"[ERROR] LLM QA chain failed: {e}")
            return "Sorry, there was a problem generating an answer."
    except Exception as e:
        print(f"[ERROR] General error in /get route: {e}")
        return "Sorry, an unexpected error occurred. Please check the server logs."



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)