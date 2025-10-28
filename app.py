import os
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import requests
import time
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "promptshield_model")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key="
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_RETRIES = 5

# ----------------------------
# Load DistilBERT model with caching
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()

# ----------------------------
# Label mapping
# ----------------------------
label_map = {0: "jailbreak", 1: "safe", 2: "suspicious", 3: "unsafe"}

# ----------------------------
# Prompt Classification Function
# ----------------------------
def classify_prompt(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()
    return label_map[pred_idx], probs[pred_idx].item(), probs.cpu().tolist()  # Also return all probs

# ----------------------------
# Gemini API Call Function
# ----------------------------
def get_gemini_response(prompt):
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    api_url_with_key = f"{GEMINI_API_URL}{GEMINI_API_KEY}"

    retries, wait_time = 0, 1
    while retries < MAX_RETRIES:
        try:
            response = requests.post(api_url_with_key, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            candidates = result.get('candidates', [])
            if candidates:
                parts = candidates[0].get('content', {}).get('parts', [])
                if parts:
                    return parts[0].get('text', '')
            return "Error: Could not extract text from Gemini response."
        except requests.exceptions.RequestException as e:
            retries += 1
            time.sleep(wait_time)
            wait_time *= 2
        except Exception as e:
            return f"Unexpected error: {e}"
    return f"Error: Failed after {MAX_RETRIES} attempts."

# ----------------------------
# Streamlit Frontend
# ----------------------------
st.set_page_config(layout="wide", page_title="PromptShield AI🛡️")
st.title("PromptShield AI🛡️")
st.caption("Enter a prompt to test the multi-class security classifier.")

user_prompt = st.text_area("Your Prompt:", height=150, placeholder="Ask me anything...")

if st.button("Submit Prompt"):
    if user_prompt.strip():
        with st.spinner("Analyzing prompt..."):
            classification, confidence, all_probs = classify_prompt(user_prompt)

        st.info(f"**Classification:** {classification.upper()}  |  **Confidence:** {confidence:.3f}")
        st.write(f"All probabilities (label order 0..3: jailbreak, safe, suspicious, unsafe): {all_probs}")

        # Act based on classification
        if classification == "safe":
            st.success("✅ Safe prompt. Sending to LLM...")
            with st.spinner("Generating response from Gemini..."):
                response = get_gemini_response(user_prompt)
            st.markdown("**LLM Response:**")
            st.markdown(response)

        elif classification == "unsafe":
            st.error("❌ UNSAFE prompt. Request blocked.")
            st.warning("This prompt appears to request harmful or malicious content.")

        elif classification == "suspicious":
            st.warning("⚠️ SUSPICIOUS prompt. Request blocked (Warning).")
            st.info("May relate to reconnaissance or info-gathering. Authorization may be required.")

        elif classification == "jailbreak":
            st.error("🚫 JAILBREAK prompt. Request blocked and flagged.")
            st.warning("This prompt attempts to bypass AI safety guidelines.")

        else:
            st.error("Error: Unknown classification result.")

    else:
        st.warning("Please enter a prompt.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Powered by PromptShield Classifier & Google Gemini")
# promptshield_app.py
