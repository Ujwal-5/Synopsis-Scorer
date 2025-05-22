# Synopsis Scorer with Privacy Protection

This application evaluates the quality of text synopses against their source content while maintaining privacy through robust text anonymization techniques.

## Features

- **Synopsis Quality Assessment**: Scores synopses based on content coverage, clarity, and coherence
- **Privacy Protection**: Anonymizes sensitive information in both source articles and synopses
- **LLM-Powered Feedback**: Provides qualitative feedback using Gemma 3 4B LLM
- **User-Friendly Interface**: Built with Streamlit for easy interaction

## Setup Instructions

### Prerequisites

- Python 3.8+
- At least 8GB RAM (recommended for LLM inference)
- 5GB disk space

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/synopsis-scorer.git
   cd synopsis-scorer
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```

5. Download the Gemma model:
   The application will automatically download the quantized Gemma model on first run


### Running the Application

1. Create a `.streamlit/secrets.toml` file with your access token:
   ```
   echo 'access_token = "your_secret_token"' > .streamlit/secrets.toml
   echo 'hf_token = "your_huggingface_token"' > .streamlit/secrets.toml
   ```

2. Start the application:
   ```
   streamlit run app.py
   ```

3. Open your browser and go to `http://localhost:8501`

## Usage

1. Enter the access token to unlock the application
2. Upload an article file (PDF or TXT format)
3. Upload a synopsis file (TXT format)
4. Click "Evaluate" to process and score the synopsis
5. Review the scoring metrics and LLM feedback

## Project Structure

```
synopsis-scorer/
├── app.py                  # Main Streamlit application
├── utils.py                # Utilities for text processing and scoring
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
├── .streamlit/
│   └── secrets.toml        # Configuration secrets
```

