import streamlit as st
from utils import extract_text, anonymize_text, score_synopsis
from llama_cpp import Llama
import os
from huggingface_hub import snapshot_download
from huggingface_hub import login


st.set_page_config(page_title="Synopsis Scorer", layout="wide")

# --- Access Control ---
TOKEN = st.secrets.get("access_token")
user_token = st.text_input("Enter Access Token to Continue", type="password")
if user_token != TOKEN:
    st.warning("Please enter a valid access token.")
    st.stop()

# --- Hugging Face Token Configuration ---
hf_token = st.secrets.get("hf_token") if "hf_token" in st.secrets else os.environ.get("HF_TOKEN")
if not hf_token and not os.path.exists("models/gemma-3-4b-it-q4_0.gguf"):
    st.warning("Hugging Face token not found. Please add it to your secrets or environment variables.")
    hf_token = st.text_input("Enter your Hugging Face token:", type="password")

login(hf_token)

print("Looking for model at:", os.path.abspath("gemma-3-4b-it-q4_0.gguf"))

# Choose a directory to store the model
model_dir = "./gemma-3-4b-it-qat-q4_0"

# Download the GGUF model
snapshot_download(
    repo_id="google/gemma-3-4b-it-qat-q4_0-gguf",
    local_dir=model_dir,
    local_dir_use_symlinks=False  # Ensures real files are written, not symlinks
)

# --- File Upload ---
st.title("📘 Synopsis Scorer with Privacy Protection")
article_file = st.file_uploader("Upload the Article (.pdf/.txt)", type=["pdf", "txt"])
synopsis_file = st.file_uploader("Upload the Synopsis (.txt)", type=["txt"])

if article_file and synopsis_file:
    with st.spinner("Reading files..."):
        article = extract_text(article_file)
        synopsis = extract_text(synopsis_file)

    st.subheader("Preview")
    st.text_area("Article", article[:1000] + "...", height=200)
    st.text_area("Synopsis", synopsis, height=150)

    if st.button("Evaluate"):
        with st.spinner("Scoring..."):
            scores = score_synopsis(article, synopsis)

            # Anonymization
            article_anon = anonymize_text(article)
            synopsis_anon = anonymize_text(synopsis)
            # Estimate n_ctx
            total_text = article_anon + synopsis_anon
            estimated_tokens = int(len(total_text)/3.5)
            n_ctx = estimated_tokens + 500
            
            article_limit = 80000 # max_article_chars = 32,000 tokens×3.5 (approx_chars_per_token)≈112,000 characters; 112,000 - 32000(space for synopsis)= 80000
            
            # LLM feedback
            try:
                llm = Llama(
                model_path="./gemma-3-4b-it-qat-q4_0/gemma-3-4b-it-q4_0.gguf",
                    n_ctx=n_ctx,
                    n_threads=2,
                    n_batch=128
                )
                
                prompt = (
                    "You are an expert writing evaluator. The user has uploaded two text documents: "
                    "1) a short synopsis, and 2) a longer article (source content). "
                    "Without copying or storing the full content, analyze the synopsis and evaluate its quality in comparison to the article. "
                    "Assess it on the basis of relevance, coverage, clarity, and coherence.\n\n"
                    "Return:\n- A score out of 100\n- 2 to 3 lines of qualitative feedback\n\n"
                    f"Here is the source article:\n{article_anon[:article_limit]}\n\nHere is the synopsis:\n{synopsis_anon}"
                )
                result = llm.create_chat_completion(messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}])
                feedback = result["choices"][0]["message"]["content"]
            except Exception as e:
                feedback = "LLM feedback not available: " + str(e)

        st.success("Evaluation Complete ✅")

        st.metric("Total Score", f"{scores['total']} / 100")
        st.progress(scores["total"] / 100)

        st.subheader("Score Breakdown")
        st.write(f"📘 Content Coverage: {scores['content_coverage']} / 50")
        st.write(f"🧠 Clarity: {scores['clarity']} / 25")
        st.write(f"🔗 Coherence: {scores['coherence']} / 25")

        st.subheader("LLM Feedback")
        st.write(feedback)
