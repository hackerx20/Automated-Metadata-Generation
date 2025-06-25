# Automated Metadata Generation System

A comprehensive Streamlit-based web application that automatically generates semantic metadata from uploaded documents using advanced Natural Language Processing (NLP) techniques.

![System Architecture](https://img.shields.io/badge/Framework-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20NLTK-green)

## 🎯 Overview

This system processes PDF, DOCX, and TXT files to extract meaningful metadata including keywords, named entities, topics, sentiment analysis, and document classification. It provides an intuitive web interface for document upload and comprehensive metadata visualization.

## ✨ Key Features

- **Multi-format Document Processing**: Supports PDF, DOCX, and TXT files
- **OCR Integration**: Handles scanned PDFs using Tesseract OCR
- **Advanced NLP Analysis**: 
  - Named Entity Recognition (NER)
  - Keyword extraction using YAKE algorithm
  - Sentiment analysis with VADER
  - Topic modeling and classification
  - Readability metrics calculation
- **Interactive Dashboard**: Real-time processing status and results visualization
- **Export Functionality**: Download metadata in JSON and CSV formats
- **Processing History**: Track and review previously processed documents

## 🏗️ System Architecture

```
Frontend (Streamlit Web App)
    ↓
Backend Processing Pipeline
    ↓
Document Processing → Content Extraction → Semantic Analysis → Metadata Generation
    ↓
Output (Structured Metadata + Visualization)
```

## 📋 Requirements

### System Dependencies
- Python 3.11+
- Tesseract OCR (for scanned PDF processing)

### Python Packages
See `project_requirements.txt` for the complete list of dependencies.

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd automated-metadata-generation
```

2. Install Python dependencies:
```bash
pip install -r project_requirements.txt
```

3. Download required NLP models:
```bash
python -m spacy download en_core_web_sm
```

4. Install NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
```

## 🚀 Usage

### Running the Application

1. Start the Streamlit server:
```bash
streamlit run app.py --server.port 5000
```

2. Open your browser and navigate to `http://localhost:5000`

### Using the System

1. **Upload Document**: Use the file uploader to select a PDF, DOCX, or TXT file (max 10MB)
2. **Process Document**: Click "Process Document" to start analysis
3. **View Results**: Navigate through the tabs to explore generated metadata:
   - Overview: Basic file information and summary
   - Content Analysis: Text statistics and readability metrics
   - Semantic Analysis: Keywords, entities, and topics
   - Statistics: Visual analytics and charts
   - Document Preview: Text preview of processed content
4. **Export Data**: Download results in JSON or CSV format

## 📊 Metadata Schema

The system generates structured metadata following this schema:

```json
{
  "metadata_schema": {
    "version": "1.0",
    "generator": "Automated Metadata Generation System",
    "generated_at": "ISO_timestamp",
    "metadata_id": "unique_id"
  },
  "file_info": {
    "filename": "string",
    "file_size": "number",
    "file_type": "string",
    "upload_timestamp": "datetime"
  },
  "content_analysis": {
    "word_count": "number",
    "character_count": "number",
    "language": "string",
    "readability_score": "number",
    "sentiment_score": "number"
  },
  "semantic_analysis": {
    "keywords": ["array"],
    "key_phrases": ["array"],
    "named_entities": ["array"],
    "topics": ["array"],
    "summary": "string"
  }
}
```

## 🔧 Configuration

The system can be configured through `config/config.yaml`:

- File processing settings (max size, supported formats)
- NLP model configurations
- UI customization options
- Export format settings
- Security parameters

## 📁 Project Structure

```
automated-metadata-generation/
├── app.py                          # Main Streamlit application
├── project_requirements.txt        # Python dependencies
├── README.md                      # Project documentation
├── src/
│   ├── __init__.py
│   ├── document_processor.py      # Document handling functions
│   ├── content_analyzer.py        # NLP and analysis functions
│   ├── metadata_generator.py      # Metadata creation functions
│   └── utils.py                   # Utility functions
├── config/
│   └── config.yaml                # Configuration settings
├── .streamlit/
│   └── config.toml                # Streamlit configuration
└── temp/                          # Temporary file storage
```

## 🧠 NLP Pipeline

1. **Text Extraction**: Extract text content using format-specific methods
2. **Preprocessing**: Clean and normalize text data
3. **Basic Analysis**: Calculate word count, sentences, paragraphs
4. **Language Detection**: Identify document language
5. **Readability Assessment**: Calculate Flesch reading ease scores
6. **Sentiment Analysis**: Determine document sentiment using VADER
7. **Keyword Extraction**: Extract key terms using YAKE algorithm
8. **Named Entity Recognition**: Identify persons, organizations, locations
9. **Topic Modeling**: Classify document topics and themes
10. **Document Classification**: Categorize document type and purpose

## 🎨 User Interface

The application features a clean, intuitive interface with:

- **Upload Interface**: Drag-and-drop file upload with validation
- **Processing Status**: Real-time progress indicators
- **Tabbed Results**: Organized metadata display
- **Visual Analytics**: Charts and graphs for data insights
- **Export Options**: Multiple download formats
- **Processing History**: Track previous analyses

## 🔒 Security

- File type validation and size limits
- Filename sanitization
- Temporary file cleanup
- Safe text processing
- Input validation throughout the pipeline

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 5000
```

### Production Deployment
The application is configured for deployment on cloud platforms with:
- Streamlit configuration in `.streamlit/config.toml`
- Environment-specific settings
- Health check endpoints
- Static file serving optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🔮 Future Enhancements

- Multi-language support
- Advanced topic modeling with transformers
- Batch processing capabilities
- API endpoint development
- Database integration for metadata storage
- Custom classification model training
- Real-time collaborative features