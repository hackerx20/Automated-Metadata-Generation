import os
import hashlib
import mimetypes
from typing import Tuple, Optional
import streamlit as st

def validate_file(uploaded_file) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file for type, size, and basic security checks
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check if file has required attributes
        if not hasattr(uploaded_file, 'size') or not hasattr(uploaded_file, 'name') or not hasattr(uploaded_file, 'type'):
            return False, "Invalid file object"
        
        # Check file size (5MB limit - reduced from 10MB to avoid upload issues)
        max_size = 5 * 1024 * 1024  # 5MB
        if uploaded_file.size and uploaded_file.size > max_size:
            return False, f"File size ({format_file_size(uploaded_file.size)}) exceeds maximum allowed size ({format_file_size(max_size)})"
        
        # Check if file is empty
        if uploaded_file.size == 0:
            return False, "File is empty"
        
        # Check file extension first (more reliable than MIME type)
        filename = uploaded_file.name.lower() if uploaded_file.name else ""
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
        
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            return False, f"File extension not supported. Allowed extensions: {', '.join(allowed_extensions)}"
        
        # Basic security check - filename
        if uploaded_file.name and any(char in uploaded_file.name for char in ['..', '/', '\\', '<', '>', '|', ':', '*', '?', '"']):
            return False, "Filename contains invalid characters"
        
        # Check file type (more lenient)
        allowed_types = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/plain',
            'application/octet-stream'  # Sometimes files are detected as this
        ]
        
        if uploaded_file.type and uploaded_file.type not in allowed_types:
            # Still allow if extension is correct
            if not any(filename.endswith(ext) for ext in allowed_extensions):
                return False, f"File type '{uploaded_file.type}' is not supported. Allowed types: PDF, DOCX, TXT"
        
        return True, None
        
    except Exception as e:
        return False, f"File validation error: {str(e)}"

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def get_file_hash(file_content: bytes) -> str:
    """
    Generate SHA-256 hash of file content
    
    Args:
        file_content: File content as bytes
        
    Returns:
        SHA-256 hash string
    """
    return hashlib.sha256(file_content).hexdigest()

def detect_mime_type(filename: str) -> str:
    """
    Detect MIME type from filename
    
    Args:
        filename: Name of the file
        
    Returns:
        MIME type string
    """
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        return mime_type
    
    # Fallback mapping
    extension = os.path.splitext(filename.lower())[1]
    extension_mapping = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.txt': 'text/plain'
    }
    
    return extension_mapping.get(extension, 'application/octet-stream')

def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing or replacing unsafe characters
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    safe_name = filename
    
    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove multiple underscores
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    
    # Remove leading/trailing underscores and dots
    safe_name = safe_name.strip('._')
    
    # Ensure filename is not empty
    if not safe_name:
        safe_name = 'unnamed_file'
    
    return safe_name

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def clean_text_for_display(text: str) -> str:
    """
    Clean text for display in UI (remove excessive whitespace, etc.)
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Replace multiple whitespace with single space
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def create_download_filename(original_filename: str, suffix: str, extension: str) -> str:
    """
    Create filename for download with suffix and extension
    
    Args:
        original_filename: Original file name
        suffix: Suffix to add (e.g., '_metadata')
        extension: File extension (e.g., 'json')
        
    Returns:
        Download filename
    """
    # Remove original extension
    name_without_ext = os.path.splitext(original_filename)[0]
    
    # Clean the name
    clean_name = safe_filename(name_without_ext)
    
    # Add suffix and extension
    return f"{clean_name}{suffix}.{extension}"

def format_timestamp(timestamp_str: str) -> str:
    """
    Format ISO timestamp for display
    
    Args:
        timestamp_str: ISO format timestamp string
        
    Returns:
        Formatted timestamp
    """
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

def get_file_info_summary(file_info: dict) -> str:
    """
    Create a summary string from file info
    
    Args:
        file_info: File information dictionary
        
    Returns:
        Summary string
    """
    filename = file_info.get('filename', 'Unknown')
    size = format_file_size(file_info.get('file_size', 0))
    file_type = file_info.get('file_type', 'Unknown')
    
    return f"{filename} ({size}, {file_type})"

def validate_metadata_completeness(metadata: dict) -> dict:
    """
    Validate metadata completeness and return status
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Validation status dictionary
    """
    required_fields = [
        'file_info.filename',
        'content_analysis.word_count',
        'semantic_analysis.keywords'
    ]
    
    validation_status = {
        'is_complete': True,
        'missing_fields': [],
        'warnings': []
    }
    
    def get_nested_value(data, path):
        keys = path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    for field in required_fields:
        value = get_nested_value(metadata, field)
        if value is None or (isinstance(value, (list, str)) and len(value) == 0):
            validation_status['missing_fields'].append(field)
            validation_status['is_complete'] = False
    
    # Check for warnings
    if metadata.get('content_analysis', {}).get('word_count', 0) < 100:
        validation_status['warnings'].append('Document has very few words, metadata quality may be limited')
    
    if not metadata.get('semantic_analysis', {}).get('keywords'):
        validation_status['warnings'].append('No keywords extracted, consider checking document content')
    
    return validation_status

# Streamlit utility functions
def display_success_message(message: str):
    """Display success message with icon"""
    st.success(f"✅ {message}")

def display_error_message(message: str):
    """Display error message with icon"""
    st.error(f"❌ {message}")

def display_warning_message(message: str):
    """Display warning message with icon"""
    st.warning(f"⚠️ {message}")

def display_info_message(message: str):
    """Display info message with icon"""
    st.info(f"ℹ️ {message}")

def create_metric_card(title: str, value: str, delta: Optional[str] = None):
    """Create a metric display card"""
    st.metric(label=title, value=value, delta=delta)

def create_download_button(data: str, filename: str, mime_type: str, button_text: str):
    """Create a download button"""
    return st.download_button(
        label=button_text,
        data=data,
        file_name=filename,
        mime=mime_type
    )
