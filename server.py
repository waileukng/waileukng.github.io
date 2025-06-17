from flask import Flask, request, jsonify, send_from_directory
import os
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='public', static_url_path='')

# Load environment variables
load_dotenv()
XAI_API_KEY = os.getenv('XAI_API_KEY')
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not found in environment variables. Ensure .env file exists with XAI_API_KEY.")

# File paths
CV_PATH = 'xlsx_output/cv.txt'
XLSX_PATH = 'xlsx_output/project_data.xlsx'

# Verify file existence
if not os.path.exists(CV_PATH):
    raise FileNotFoundError(f"cv.txt not found at {CV_PATH}")
if not os.path.exists(XLSX_PATH):
    raise FileNotFoundError(f"project_data.xlsx not found at {XLSX_PATH}")

# Load and process cv.txt with encoding fallback
try:
    loader = TextLoader(CV_PATH, encoding='utf-8')
    documents = loader.load()
except UnicodeDecodeError:
    logger.warning(f"UTF-8 decoding failed for {CV_PATH}, trying latin1 encoding")
    loader = TextLoader(CV_PATH, encoding='latin1')
    documents = loader.load()
except Exception as e:
    raise RuntimeError(f"Failed to load cv.txt: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Further reduced for short document
    chunk_overlap=20,  # Reduced for efficiency
    length_function=len,
    add_start_index=True
)
chunks = text_splitter.split_documents(documents)
logger.info(f"Created {len(chunks)} chunks from cv.txt")

# Set up FAISS vector store with lightweight HuggingFace embeddings
try:
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Force CPU for Render
    )
    db = FAISS.from_documents(chunks, embedding)
    logger.info(f"Initialized FAISS with {len(chunks)} chunks")
except Exception as e:
    raise RuntimeError(f"Failed to initialize FAISS vector store: {e}")

# Prompt template
PROMPT_TEMPLATE = """
You are William NG's AI assistant, answering questions based on his CV and project data.
Use the following context and project data to provide concise, accurate responses.

CV Context:
{context}

Project Data (from XLSX):
{project_data}

---

Question: {question}
Answer:
"""

# Load XLSX data
def load_file_data(filename):
    try:
        df = pd.read_excel(filename)
        csv_data = df.to_csv(index=False)
        return csv_data[:200]  # Further reduced for memory
    except Exception as e:
        logger.error(f'XLSX Error: {e}')
        return ''

# Grok API call
def call_grok_api(prompt, max_tokens=80, temperature=0.7):  # Further reduced max_tokens
    try:
        headers = {
            'Authorization': f'Bearer {XAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'grok-3',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        response = requests.post('https://api.x.ai/v1/chat/completions', headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Grok API error: {e}")
        return None

@app.route('/')
def serve_index():
    return send_from_directory('public', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('message')
    if not query:
        return jsonify({'error': 'No message provided'}), 400

    try:
        project_data = load_file_data(XLSX_PATH)
        results = db.similarity_search_with_relevance_scores(query, k=3)
        if not results or results[0][1] < 0.5:  # Lowered threshold for better recall
            return jsonify({'reply': 'Sorry, I couldn’t find relevant information to answer your question.'})

        context = '\n\n---\n\n'.join([doc.page_content for doc, score in results])
        prompt = PROMPT_TEMPLATE.format(context=context, project_data=project_data, question=query)

        response = call_grok_api(prompt)
        if response is None:
            return jsonify({'error': 'Failed to get response from Grok API'}), 500

        sources = [doc.metadata.get('source', None) for doc, _ in results]
        return jsonify({'reply': response, 'sources': sources})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Failed to process message'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))  # Use Render's PORT
    app.run(host='0.0.0.0', port=port, debug=False)  # Disable debug
