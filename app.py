import streamlit as st
from logic import get_desi_recipe, analyze_image
from PIL import Image
import io

# 1. Page Config & Professional CSS
st.set_page_config(page_title="DesiVision AI", page_icon="🥘", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f9fbff; }
    .main-card {
        background-color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 12px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar Stats & Controls
with st.sidebar:
    st.title("👨‍🍳 Chef Settings")
    diet_choice = st.selectbox("Diet Preference", ["all", "vegetarian", "non vegetarian"])
    st.divider()
    st.metric("Recipes Indexed", "255+", "Healthy")
    st.info("System: Llama 3 + Llava (Vision) + ChromaDB")

# 3. Main Dashboard Header
st.title("🥘DesiVision AI")
st.caption("Identify ingredients via Photo or Text and get authentic Indian recipes.")

# 4. Layout: Vision vs Text
col1, col2 = st.columns([1, 1], gap="large")

detected_text = ""

with col1:
    st.subheader(" Camera Input")
    uploaded_file = st.file_uploader("Snap a photo of your ingredients", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)
        if st.button("🔍 Scan Image"):
            with st.spinner("AI is analyzing the photo..."):
                img_bytes = uploaded_file.getvalue()
                detected_text = analyze_image(img_bytes)
                st.success(f"Detected: {detected_text}")

with col2:
    st.subheader("📝 Ingredient List")
    # If vision detected something, it fills this box automatically
    final_ingredients = st.text_area("What do you have?", value=detected_text, height=150, placeholder="e.g. Potato, Onion, Cumin")
    
    if st.button("🚀 Generate Best Recipe", type="primary"):
        if final_ingredients:
            with st.status("👨‍🍳 Chef is thinking...", expanded=True) as status:
                st.write("Filtering database...")
                ing_list = [i.strip() for i in final_ingredients.split(",")]
                recipe = get_desi_recipe(ing_list, diet_choice)
                status.update(label="Recipe Found!", state="complete")
            
            # Display Final Result
            st.markdown("### 🍴 Chef's Masterpiece")
            with st.container(border=True):
                st.markdown(recipe)
        else:
            st.warning("Please upload a photo or type ingredients.")

st.markdown("---")
st.caption("Master's Level Portfolio Project | Powered by Local RAG Architecture")