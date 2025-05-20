# Synopsis Scorer: Methodology and Privacy Protection Strategy

## Scoring Methodology

The synopsis quality evaluation system employs a multi-dimensional approach to assess how effectively a synopsis captures and communicates the content of its source article:

### Components (Total: 100 points)

#### 1. **Content Coverage – 50 points**
- **What it checks:** How well the synopsis captures the key ideas from the article.
- **How it's done:**
  - Uses the `all-MiniLM-L6-v2` SentenceTransformer model to convert both texts into vector form.
  - Calculates the cosine similarity between the article and synopsis embeddings (range: 0 to 1).
- **Scoring:** `similarity × 50` (e.g., 0.9 similarity = 45 points).
- **Why it matters:** A strong synopsis reflects the main points from the original content.

#### 2. **Clarity – 25 points**
- **What it checks:** How clearly and precisely the synopsis is written.
- **How it's done:**
  - Calculates lexical diversity: `(unique words / total words) × 25`.
- **Why it matters:** More diverse vocabulary indicates better language use and avoids repetition.

#### 3. **Coherence – 25 points**
- **What it checks:** How logically the synopsis is structured and whether it flows smoothly.
- **How it's done:**
  - Gives 5 points per sentence, up to a maximum of 5 sentences.
- **Why it matters:** Clear, well-structured writing is easier to understand and follow.

### Advanced Feedback

In addition to the quantitative scoring, the system leverages the Gemma 3 4B LLM to provide qualitative feedback on synopsis quality. The model is guided through careful prompt engineering to focus on relevance, coverage, clarity, and coherence without storing or reproducing the original text content.

## Privacy Protection Strategy

The system implements a comprehensive data privacy approach to protect sensitive information:

### Multi-Layer Anonymization

1. **Named Entity Recognition**
   - Uses spaCy's NER capabilities to identify and replace sensitive entities:
     - PERSON: Individual names
     - DATE: Temporal identifiers
     - LOCATION/GPE: Geographic references
     - ORG: Organization names

2. **Regex Pattern Matching**
   - Supplements NER with custom regular expressions to catch:
     - Email addresses
     - URLs and web links
     - Phone numbers
     - Identification numbers/codes

3. **Privacy-Preserving LLM Integration**
   - Applies anonymization before sending text to the LLM
   - Uses a quantized model running locally to avoid data transmission to external APIs
   - Implements character limits to prevent overloading and potential information leakage

### System Design Considerations

- **Local Processing**: All text processing occurs on the user's machine
- **Access Control**: Token-based authentication restricts unauthorized access
- **Data Minimization**: Preview displays only limited text portions
- **Secure LLM Integration**: Carefully constructed prompts instruct the LLM to analyze without storing or reproducing sensitive content

This privacy-first approach ensures that the system can provide valuable evaluation feedback while maintaining the confidentiality of sensitive information in both source articles and synopses.