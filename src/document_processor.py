import os
import re
import logging
from typing import Dict, Optional, Tuple
import PyPDF2
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Document processing module for extracting text from various file formats
    """
    
    def __init__(self):
        self.supported_formats = ['pdf', 'docx', 'txt']
    
    def extract_text(self, file_path: str, file_type: str) -> str:
        """
        Extract text from document based on file type
        
        Args:
            file_path: Path to the document file
            file_type: MIME type of the file
            
        Returns:
            Extracted text content
        """
        try:
            if 'pdf' in file_type.lower():
                return self.extract_text_from_pdf(file_path)
            elif 'word' in file_type.lower() or 'docx' in file_type.lower():
                return self.extract_text_from_docx(file_path)
            elif 'text' in file_type.lower() or 'txt' in file_type.lower():
                return self.extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file with OCR fallback
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        
        try:
            # First try with pdfplumber (better text extraction)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # If pdfplumber didn't extract much text, try PyPDF2
            if len(text.strip()) < 100:
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            
            # If still no text, try OCR (for scanned PDFs)
            if len(text.strip()) < 50:
                text = self._ocr_pdf(file_path)
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            # Try OCR as last resort
            try:
                text = self._ocr_pdf(file_path)
            except:
                raise Exception(f"Failed to extract text from PDF: {str(e)}")
        
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from DOCX file
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text content
        """
        try:
            doc = Document(file_path)
            text = ""
            
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text += cell.text + " "
                    text += "\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            raise
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """
        Extract text from TXT file with encoding detection
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            Extracted text content
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, read as binary and decode with errors='ignore'
            with open(file_path, 'rb') as file:
                return file.read().decode('utf-8', errors='ignore')
                
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {str(e)}")
            raise
    
    def _ocr_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF using OCR (for scanned documents)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            OCR extracted text
        """
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Convert page to image
                    page_image = page.to_image(resolution=300)
                    
                    # Convert PIL image to format pytesseract can use
                    img_bytes = io.BytesIO()
                    page_image.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    # Perform OCR
                    page_text = pytesseract.image_to_string(
                        Image.open(img_bytes),
                        config='--psm 3'  # Page segmentation mode
                    )
                    
                    if page_text.strip():
                        text += page_text + "\n"
            
            return text
            
        except Exception as e:
            logger.warning(f"OCR failed for {file_path}: {str(e)}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned and preprocessed text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=\<\>\|\~\`]', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Clean up common OCR errors
        text = re.sub(r'\b[A-Z]{1}\s+(?=[A-Z])', '', text)  # Remove single letters
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)  # Fix spacing before punctuation
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        # Strip and return
        return text.strip()
    
    def detect_file_type(self, file_path: str) -> str:
        """
        Detect file type based on extension and content
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected file type
        """
        _, ext = os.path.splitext(file_path.lower())
        
        type_mapping = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.txt': 'text/plain'
        }
        
        return type_mapping.get(ext, 'unknown')
    
    def get_document_structure(self, text: str) -> Dict:
        """
        Identify document structure (headings, sections, etc.)
        
        Args:
            text: Document text
            
        Returns:
            Dictionary containing structure information
        """
        structure = {
            'headings': [],
            'sections': [],
            'paragraphs': [],
            'has_toc': False
        }
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Detect headings (lines that are short, capitalized, or numbered)
            if self._is_heading(line):
                structure['headings'].append({
                    'text': line,
                    'line_number': i,
                    'level': self._get_heading_level(line)
                })
            
            # Detect sections
            if len(line.split()) < 10 and any(word in line.lower() for word in 
                                           ['chapter', 'section', 'part', 'introduction', 'conclusion']):
                structure['sections'].append({
                    'text': line,
                    'line_number': i
                })
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        structure['paragraphs'] = paragraphs[:10]  # First 10 paragraphs
        
        # Check for table of contents
        structure['has_toc'] = self._detect_toc(text)
        
        return structure
    
    def _is_heading(self, line: str) -> bool:
        """Check if a line is likely a heading"""
        if len(line) > 100:  # Too long to be a heading
            return False
        
        # Check for numbered headings
        if re.match(r'^\d+\.?\s+[A-Z]', line):
            return True
        
        # Check for all caps headings
        if line.isupper() and len(line.split()) <= 8:
            return True
        
        # Check for title case headings
        words = line.split()
        if len(words) <= 8 and sum(1 for w in words if w[0].isupper() if w) > len(words) * 0.7:
            return True
        
        return False
    
    def _get_heading_level(self, line: str) -> int:
        """Determine heading level (1-6)"""
        # Check for numbered headings
        match = re.match(r'^(\d+)\.', line)
        if match:
            return min(int(match.group(1)), 6)
        
        # Based on length and formatting
        if line.isupper():
            return 1
        elif len(line.split()) <= 3:
            return 2
        else:
            return 3
    
    def _detect_toc(self, text: str) -> bool:
        """Detect if document has a table of contents"""
        toc_indicators = [
            'table of contents', 'contents', 'index',
            r'\d+\s+\.\.\.\s+\d+',  # Page numbers with dots
            r'chapter \d+.*\d+$'     # Chapter with page number
        ]
        
        for indicator in toc_indicators:
            if re.search(indicator, text.lower(), re.MULTILINE):
                return True
        
        return False
