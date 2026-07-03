import os
import json
import string
import re
import numpy as np
import streamlit as st
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Sherlock's Echo | Next Word Predictor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "next_word_model.keras")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# Custom CSS for premium design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Fira+Code:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header & Titles */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .header-accent {
        background: linear-gradient(135deg, #a78bfa 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Styled Containers & Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
    }
    
    /* Input Box Custom Styling */
    .stTextInput>div>div>input {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.1rem !important;
        padding: 12px 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Custom buttons for smart keyboard suggestions */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #e2e8f0;
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        padding: 8px 16px;
        width: 100%;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: #ffffff;
        border-color: #6366f1;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
    }
    div.stButton > button:first-child:active {
        transform: translateY(0);
    }
    
    /* Code output and generated text */
    .generated-box {
        font-family: 'Fira Code', monospace;
        background-color: #0b0f19;
        border-left: 4px solid #6366f1;
        padding: 16px;
        border-radius: 0 12px 12px 0;
        color: #a7f3d0;
        margin-top: 15px;
        font-size: 1.1rem;
        line-height: 1.6;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Accent tags */
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
        background-color: rgba(99, 102, 241, 0.15);
        color: #a78bfa;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for preprocessing
def clean_and_tokenize(text, vocab_dict):
    """Lowercases, strips punctuation, splits by space, and maps to vocab IDs."""
    text = text.lower()
    # Keras TextVectorization default strip punctuation regex
    punc_pattern = re.compile(f"[{re.escape(string.punctuation)}]")
    clean_text = punc_pattern.sub("", text)
    words = clean_text.split()
    
    tokens = []
    for word in words:
        if word in vocab_dict:
            tokens.append(vocab_dict[word])
        else:
            tokens.append(1)  # Index 1 is [UNK]
    return tokens

# Cache resource loading
@st.cache_resource
def load_assets():
    # Load model
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}. Please run the Jupyter notebook training cell to export the model first!")
        st.stop()
    
    # Load vocab
    if not os.path.exists(VOCAB_PATH):
        st.error(f"Vocabulary file not found at: {VOCAB_PATH}. Please run the Jupyter notebook training cell to export the vocab first!")
        st.stop()
        
    # Load config
    max_len = 18 # fallback default
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                max_len = cfg.get("max_len", 18)
        except Exception:
            pass
            
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return model, vocab, max_len
    except Exception as e:
        st.error(f"Error loading model assets: {str(e)}")
        st.stop()

# Load model and vocabulary
model, vocab, max_len = load_assets()
vocab_dict = {word: idx for idx, word in enumerate(vocab)}

# Layout & Sidebar
st.sidebar.markdown(f"## 🔍 Model Parameters")
st.sidebar.markdown("Configure how the text generation runs:")

temp = st.sidebar.slider(
    "Temperature", 
    min_value=0.1, 
    max_value=2.0, 
    value=0.7, 
    step=0.1,
    help="Higher values = more creative/random text. Lower values = more predictable text."
)

rep_penalty = st.sidebar.slider(
    "Repetition Penalty", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Controls the likelihood of repeating the same word. Lower values penalize repeats more heavily."
)

gen_words_count = st.sidebar.slider(
    "Words to Generate", 
    min_value=1, 
    max_value=50, 
    value=15, 
    step=1,
    help="Number of words to generate in the Creative Generator tab."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.markdown(f"- **Vocabulary Size:** {len(vocab)} words")
st.sidebar.markdown(f"- **Sequence History Window:** {max_len - 1} words")
st.sidebar.markdown(f"- **Total Target Classes:** 10411 (Model Dense Output)")
st.sidebar.markdown("Based on the **Sherlock Holmes Next-Word Prediction Model** trained on Arthur Conan Doyle's stories.")

# Title Area
st.markdown('# 🔍 Sherlock\'s Echo <span class="header-accent">Next Word Predictor</span>', unsafe_allow_html=True)
st.markdown("An interactive AI text assistant powered by an LSTM recurrent neural network. Predict the next word or generate paragraphs in Doyle's unique literary style.", unsafe_allow_html=True)

# Main tabs
tab1, tab2 = st.tabs(["✨ Smart Suggestions", "✍️ Creative Writing Generator"])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3>Interactive Writing Assistant</h3>', unsafe_allow_html=True)
    st.markdown('Type below. The model will analyze your context and display the top three most likely next words. Click on any suggestion to instantly append it to your sentence.', unsafe_allow_html=True)
    
    # State Management for text box
    if "user_text" not in st.session_state:
        st.session_state.user_text = "Sherlock holmes was a"
        
    def update_text_state():
        st.session_state.user_text = st.session_state.user_text_input
        
    # User Input
    text_input_val = st.text_input(
        "Type your prompt here:",
        value=st.session_state.user_text,
        key="user_text_input",
        on_change=update_text_state,
        placeholder="Type a word or sentence..."
    )
    
    # Handle Button Clicks to append word
    def make_choice(word):
        # Clean current text space, append new word, add trailing space for next typing
        st.session_state.user_text = st.session_state.user_text.strip() + " " + word + " "
        st.session_state.user_text_input = st.session_state.user_text
        
    # Get Suggestions
    context = st.session_state.user_text.strip()
    suggestions = []
    
    if context:
        tokens = clean_and_tokenize(context, vocab_dict)
        # Pad sequence
        seq_len = max_len - 1
        if len(tokens) < seq_len:
            padded = [0] * (seq_len - len(tokens)) + tokens
        else:
            padded = tokens[-seq_len:]
            
        # Predict
        padded_input = np.array([padded], dtype=np.int32)
        with tf.device('/CPU:0'): # Ensure standard CPU execution context for Streamlit thread safety
            prediction = model(padded_input, training=False).numpy()
        probas = prediction[0][:len(vocab)]
        
        # Penalize padding & OOV tokens to avoid suggesting them
        probas[0] = -1
        probas[1] = -1
        
        # Get top 3 indices
        top_indices = np.argsort(probas)[-3:][::-1]
        for idx in top_indices:
            if idx < len(vocab) and probas[idx] > 0:
                suggestions.append(vocab[idx])
                
    # Fallback to defaults if no suggestions or empty
    if not suggestions:
        suggestions = ["the", "and", "i"]
        
    # Render suggestion buttons
    st.markdown('<p style="margin-bottom: 8px; font-weight: 600; color: #a78bfa;">Next word suggestions:</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, word in enumerate(suggestions):
        with cols[i]:
            st.button(
                f"➕ {word}", 
                key=f"suggest_{word}_{i}", 
                on_click=make_choice, 
                args=(word,)
            )
            
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3>Creative Text Generator</h3>', unsafe_allow_html=True)
    st.markdown('Enter a starting prompt and specify the number of words you want the model to generate recursively. The generator applies temperature and repetition penalty settings configured in the sidebar.', unsafe_allow_html=True)
    
    start_prompt = st.text_input(
        "Enter seed text for generation:",
        value="He had no doubt that",
        key="gen_seed_input"
    )
    
    if st.button("🚀 Generate Creative Text", use_container_width=True):
        if not start_prompt.strip():
            st.warning("Please enter some seed text to start the generation!")
        else:
            with st.spinner("Sherlock is thinking..."):
                current_text = start_prompt.strip()
                generated_words = []
                seq_len = max_len - 1
                
                for step in range(gen_words_count):
                    tokens = clean_and_tokenize(current_text, vocab_dict)
                    # Pad sequence
                    if len(tokens) < seq_len:
                        padded = [0] * (seq_len - len(tokens)) + tokens
                    else:
                        padded = tokens[-seq_len:]
                        
                    # Predict
                    padded_input = np.array([padded], dtype=np.int32)
                    with tf.device('/CPU:0'):
                        prediction = model(padded_input, training=False).numpy()
                    probas = prediction[0][:len(vocab)]
                    
                    # Apply repetition penalty for the last 3 generated words
                    for word in set(generated_words[-3:]):
                        if word in vocab_dict:
                            probas[vocab_dict[word]] *= rep_penalty
                            
                    # Re-normalize
                    probas = probas / (np.sum(probas) + 1e-10)
                    
                    # Apply temperature scaling
                    # Avoid dividing by extremely small numbers
                    effective_temp = max(temp, 0.05)
                    logits = np.log(probas + 1e-10) / effective_temp
                    exp_logits = np.exp(logits)
                    probas = exp_logits / np.sum(exp_logits)
                    
                    # Prevent padding (0) and OOV (1) from being chosen
                    probas[0] = 0
                    probas[1] = 0
                    probas = probas / (np.sum(probas) + 1e-10)
                    
                    # Sample word
                    pos = np.random.choice(len(probas), p=probas)
                    predicted_word = vocab[pos]
                    
                    generated_words.append(predicted_word)
                    current_text = f"{current_text} {predicted_word}"
                
                # Show results
                st.markdown('<h4>Generated Output:</h4>', unsafe_allow_html=True)
                st.markdown(f'<div class="generated-box">{current_text}</div>', unsafe_allow_html=True)
                
    st.markdown('</div>', unsafe_allow_html=True)

# Footer info
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #64748b; font-size: 0.85rem;">'
    'Created for Next Word Prediction &bull; Minimalist Custom Styling &bull; Optimized for Streamlit Cloud'
    '</p>', 
    unsafe_allow_html=True
)
