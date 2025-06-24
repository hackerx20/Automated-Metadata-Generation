# Deployment Guide - Automated Metadata Generation System

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r project_requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm

# Install NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

# Run application
streamlit run app.py --server.port 5000
```

### Production Deployment

#### Option 1: Docker Deployment
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY project_requirements.txt .
RUN pip install -r project_requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

EXPOSE 5000

CMD ["streamlit", "run", "app.py", "--server.port", "5000", "--server.address", "0.0.0.0"]
```

#### Option 2: Cloud Platform Deployment

**Streamlit Community Cloud:**
1. Push code to GitHub repository
2. Connect to Streamlit Community Cloud
3. Configure with `project_requirements.txt`
4. Set startup command: `streamlit run app.py`

**Heroku:**
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port \$PORT --server.address 0.0.0.0" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
heroku create your-app-name
git push heroku main
```

#### Option 3: Railway Deployment
```toml
# railway.toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"
```

## Environment Variables

```bash
# Optional configuration
STREAMLIT_SERVER_PORT=5000
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## System Requirements

**Minimum:**
- Python 3.11+
- 2GB RAM
- 1GB storage

**Recommended:**
- Python 3.11+
- 4GB RAM
- 2GB storage
- Multi-core CPU for faster processing

## Troubleshooting

**Common Issues:**

1. **spaCy model not found:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK data missing:**
   ```python
   import nltk
   nltk.download('all')
   ```

3. **Tesseract not found:**
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`
   - Windows: Download from official site

4. **Memory issues:**
   - Increase system memory
   - Reduce max file size in config
   - Process files in smaller batches