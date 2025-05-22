import re 
import fitz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy 


# Load the English NLP model and the SentenceTransformer model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    else:
        return file.read().decode("utf-8")
    
def anonymize_text(text):
    doc = nlp(text)
    #Collect spaCy-detected entities
    replacements = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            replacements.append((ent.start_char, ent.end_char, "PERSON"))
        elif ent.label_ == "DATE":
            replacements.append((ent.start_char, ent.end_char, "DATE"))
        elif ent.label_ in ["GPE", "LOC"]:
            replacements.append((ent.start_char, ent.end_char, "LOCATION"))
        elif ent.label_ == "ORG":
            replacements.append((ent.start_char, ent.end_char, "ORG"))
    
    #Add regex-based matches for things spaCy misses
    regex_patterns = [
        (r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "EMAIL"),     # Email
        (r"https?://\S+|www\.\S+", "URL"),            # URLs
        (r"\b\d{10}\b", "PHONE"),                     # 10-digit phone numbers
        (r"\b[A-Z]{2,}\d{6,}\b", "ID"),               # Generic IDs (e.g., AA123456)
    ]
    for pattern, label in regex_patterns:
        for match in re.finditer(pattern, text):
            replacements.append((match.start(), match.end(), label))
    
    replacements.sort(reverse=True)
    for start, end, replacement in replacements:
        text = text[:start] + f"[{replacement}]" + text[end:]  # Adding brackets for clarity
    
    return text

def score_synopsis(article, synopsis):
    embeddings = model.encode([article, synopsis])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    content_coverage = similarity * 50
    clarity = (len(set(synopsis.split())) / max(len(synopsis.split()), 1)) * 25
    coherence = min(25, 5 * (len(synopsis.split(".")) - 1))

    total = content_coverage + clarity + coherence
    return {
        "total": round(total, 2),
        "content_coverage": round(content_coverage, 2),
        "clarity": round(clarity, 2),
        "coherence": round(coherence, 2)
    }
