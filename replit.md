# Automated Metadata Generation System

## Overview

This is a Streamlit-based web application that automatically generates semantic metadata from uploaded documents. The system processes PDF, DOCX, and TXT files using advanced Natural Language Processing (NLP) techniques to extract meaningful information and create structured metadata output.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Provides the user interface for file uploads and metadata visualization
- **Interactive Dashboard**: Real-time processing status and results display
- **Multi-page Layout**: Wide layout with expandable sidebar for navigation and history

### Backend Architecture
- **Modular Processing Pipeline**: Separated into distinct modules for document processing, content analysis, and metadata generation
- **NLP Processing**: Utilizes spaCy, NLTK, and scikit-learn for advanced text analysis
- **OCR Integration**: Pytesseract for extracting text from image-based PDFs

### Technology Stack
- **Python 3.11**: Core runtime environment
- **Streamlit**: Web framework for the user interface
- **spaCy**: Advanced NLP processing and named entity recognition
- **NLTK**: Text processing and sentiment analysis
- **scikit-learn**: Machine learning for text classification and clustering
- **Plotly**: Interactive data visualization
- **PyPDF2/pdfplumber**: PDF text extraction
- **python-docx**: DOCX document processing

## Key Components

### Document Processor (`src/document_processor.py`)
- **Purpose**: Handles file format detection and text extraction
- **Supported Formats**: PDF, DOCX, TXT
- **Features**: OCR fallback for image-based PDFs, text preprocessing
- **Technology**: PyPDF2, pdfplumber, python-docx, pytesseract

### Content Analyzer (`src/content_analyzer.py`)
- **Purpose**: Performs semantic analysis and NLP processing
- **Features**: 
  - Named Entity Recognition (NER)
  - Keyword extraction using YAKE algorithm
  - Sentiment analysis with VADER
  - Readability metrics calculation
  - Topic modeling and clustering
- **Technology**: spaCy, NLTK, scikit-learn, textstat

### Metadata Generator (`src/metadata_generator.py`)
- **Purpose**: Creates structured metadata output
- **Features**:
  - Comprehensive metadata schema generation
  - Document classification
  - Quality metrics calculation
  - Structured JSON output
- **Output Format**: JSON with standardized schema

### Utilities (`src/utils.py`)
- **Purpose**: Common utility functions for file validation and processing
- **Features**: File type validation, size checks, security validation, hash generation

## Data Flow

1. **File Upload**: User uploads document through Streamlit interface
2. **Validation**: File type, size, and security checks performed
3. **Text Extraction**: Document processor extracts text based on file type
4. **Content Analysis**: NLP pipeline analyzes text for semantic information
5. **Metadata Generation**: Structured metadata created from analysis results
6. **Visualization**: Results displayed in interactive dashboard
7. **Export**: Metadata available for download in JSON format

## External Dependencies

### Core NLP Libraries
- **spaCy**: Requires `en_core_web_sm` model for English language processing
- **NLTK**: Downloads required datasets (punkt, stopwords, vader_lexicon)
- **scikit-learn**: For machine learning-based text analysis

### Document Processing
- **Tesseract OCR**: Required for image-based PDF text extraction
- **System Libraries**: freetype, libjpeg, libtiff, libwebp for image processing

### Visualization
- **Plotly**: Interactive charts and graphs
- **Streamlit**: Web framework with built-in visualization components

## Deployment Strategy

### Replit Configuration
- **Runtime**: Python 3.11 with Nix package management
- **Deployment Target**: Autoscale deployment
- **Port**: 5000 (configured for Streamlit server)
- **Workflow**: Parallel execution with dedicated Streamlit server task

### System Requirements
- **Memory**: Sufficient for NLP model loading (spaCy, NLTK)
- **Storage**: Temporary file storage for document processing
- **Network**: Internet access for downloading NLP models and datasets

### Configuration
- **Application Config**: `config/config.yaml` for system settings
- **Streamlit Config**: `.streamlit/config.toml` for server configuration
- **Dependencies**: `pyproject.toml` with comprehensive package listing

## Changelog

```
Changelog:
- June 24, 2025: Initial setup and complete system implementation
- June 24, 2025: Fixed NLTK punkt_tab dependency issue
- June 24, 2025: Created comprehensive project documentation and requirements file
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```