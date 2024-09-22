import os
import re

from flask import Flask, request, jsonify

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

import firebase_admin
from firebase_admin import credentials, firestore

from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI    
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from google.protobuf.json_format import MessageToDict

from log_manager import print_log

from dotenv import load_dotenv
load_dotenv()

print_log("load_key_from_env execution started")
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
print_log("load_key_from_env execution Loaded")


app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

@app.route('/upload', methods=['POST'])
def upload_document():

    print_log("upload Documentation Execution Started")
    
    # Check if the request contains a file and chat_name
    if 'file' not in request.files:
        return jsonify({"error": "File not provided"}), 400
    if 'chat-name' not in request.form:
        return jsonify({"error": "chat_name not provided"}), 400

    pdf_file = request.files['file']
    chat_name = request.form['chat-name']  # Ensure correct key here

    index_name = chat_name

    # Check if the index exists, if not create it
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=768,  # Correct dimension for the embedding model
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    print_log(pdf_file)
    
    # Extract text from the uploaded PDF using PyPDF2
    pdfreader = PdfReader(pdf_file)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # Split the extracted text into manageable chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    print_log("Emmebedding Started")

    # Initialize Google Generative AI Embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")

    # Generate embeddings for each text chunk
    embeddings = embedding_model.embed_documents(texts)

    print_log("Emmebedding model")

    # Get Pinecone index
    index = pc.Index(index_name)
    vectors = []
    for i, (embedding, text) in enumerate(zip(embeddings, texts)):
        vectors.append({
            "id": f"{chat_name}_{i}",  # Unique ID for each vector
            "values": embedding,  # Embedding vector
            "metadata": {"text": text}  # Text chunk stored as metadata
        })
    
    # Upsert vectors to Pinecone
    upsert_response = index.upsert(vectors=vectors)
    upserted_count = upsert_response.upserted_count

    print_log("Emmebedding Completed")

    print_log("Firestore Started")

    # Store the embeddings and text in Firestore
    doc_ref = db.collection('documents').document(chat_name)
    doc_ref.set({
        'chat_name': chat_name,
        'upserted_count': upserted_count,
        'document_text': texts,
    })

    print_log("Firestore ended")

    print_log("upload Documentation Execution Completed")

    return jsonify({
        "message": "Document uploaded and indexed successfully",
        "upserted_count": upserted_count,
    }), 200
    
    

@app.route('/query', methods=['POST'])
def query_document():
    chat_name = request.form.get('chat-name')  # Ensure correct key here
    question = request.form.get('question')

    if not chat_name or not question:  # Validate that both fields are present
        return jsonify({'error': 'Both chat-name and question are required'}), 400

    if not validate_question(question):
        return jsonify({'error': 'Invalid question'}), 400

    index_name = chat_name
    index = pc.Index(index_name)

    # Retrieve metadata from Firestore
    doc_ref = db.collection('documents').document(chat_name)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({'error': 'Chat name not found'}), 404

    # Initialize Google Generative AI Embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    embeddings = embedding_model.embed_query(question)

    retriever = index.query(vector=embeddings, top_k=5, include_values=False, include_metadata=True)

    documents = [Document(page_content=match['metadata']['text'], metadata=match['metadata']) for match in retriever['matches']]
    
    return user_input(retriever=documents, question=question)


def get_conversational_chain():
    prompt_template = """You are a Chatbot,
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, don't provide the wrong answer\n\n
    If the question is invalid (e.g., non-sensical, offensive, or out-of-scope), return a response indicating that the question cannot be processed.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                   temperature=0.3, api_key=GEMINI_API_KEY)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(retriever, question):
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": retriever, "question": question},
        return_only_outputs=True
    )

    return jsonify({
        "Bot": response["output_text"]  # Return the response from the chatbot
    }), 200


def validate_question(question):
    # Check if the input is empty or too short
    if not question or len(question.strip()) < 1:
        return False, "The question is too short. Please provide more detail."

    # Check for nonsensical input (using regex for gibberish)
    if re.match(r"^[a-zA-Z0-9]{10,}$", question):
        return False, "The question appears to be nonsensical. Please ask a meaningful question."

    # If all checks pass, the question is valid
    return True


if __name__ == '__main__':
    app.run(debug=True)
