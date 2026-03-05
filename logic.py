import os
import platform
import pandas as pd
import io
import base64
import streamlit as st
from PIL import Image

# 1. Cloud-Specific SQLite Fix (Must be at the very top)
if platform.system() != "Windows":
    try:
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 2. API Key Configuration
# Uses Streamlit Secrets for Cloud, or manual string for local testing
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE" # Put your key here for VS Code testing

# 3. Initialize AI Models
# Text Model: Llama 3.1 8B (Fast & Free)
llm = ChatGroq(
    temperature=0.1, 
    groq_api_key=GROQ_API_KEY, 
    model_name="llama-3.1-8b-instant"
)

# Vision Model: Llama 4 Scout (2026 Standard)
vision_llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
    groq_api_key=GROQ_API_KEY
)

# Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Vector Database Logic
def initialize_vector_db():
    # Folder-aware path for GitHub ('data' folder)
    csv_path = os.path.join("data", "indian_food.csv")
    
    if not os.path.exists(csv_path):
        st.error(f"CSV not found at {csv_path}. Check your 'data' folder on GitHub.")
        return None

    df = pd.read_csv(csv_path)
    docs = []
    for _, r in df.iterrows():
        name = r.get('name', 'Unknown')
        ingredients = r.get('ingredients', '')
        diet = str(r.get('diet', 'all')).lower().strip()
        
        text = f"Dish: {name}. Ingredients: {ingredients}"
        metadata = {"name": name, "ingredients_list": ingredients, "diet": diet}
        docs.append(Document(page_content=text, metadata=metadata))
    
    return Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )

# Load or Create DB
if os.path.exists("./chroma_db"):
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    vector_db = initialize_vector_db()

# 5. Vision Analysis Function
def analyze_image(image_bytes):
    """Processes image and identifies ingredients via Llama 4 Scout"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert RGBA to RGB for JPEG compatibility
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        # Resize to prevent 413 Payload Error
        img.thumbnail((800, 800))
        
        # Compress and Encode
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "List only the food ingredients you see in this image. Format as a comma-separated list."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }]
        
        response = vision_llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# 6. Recipe Retrieval Function
def get_desi_recipe(user_ingredients, diet_filter="all"):
    """Finds best matching Indian dish and generates a recipe"""
    query = f"Indian dish made with {', '.join(user_ingredients)}"
    
    search_kwargs = {"k": 2}
    if diet_filter != "all":
        search_kwargs["filter"] = {"diet": diet_filter}
    
    results = vector_db.similarity_search(query, **search_kwargs)
    
    context = ""
    for res in results:
        context += f"- {res.metadata['name']} (Ingredients: {res.metadata['ingredients_list']})\n"

    prompt = f"""
    You are an expert Desi Chef. A student has: {user_ingredients}.
    Best matches from database:
    {context}
    
    Task:
    1. **Reasoning**: Explain why the match is best.
    2. **Recipe**: Provide a simple step-by-step student-friendly recipe.
    3. **Student Hack**: Give one clever shortcut.
    
    Format with Bold Headers.
    """
    
    return llm.invoke(prompt).content