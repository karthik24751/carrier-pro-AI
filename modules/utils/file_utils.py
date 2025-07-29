import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_extension(filename):
    """Get the file extension from a filename."""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

def ensure_directory(directory):
    """Ensure a directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_safe_filename(filename):
    """Get a secure filename for storage."""
    return secure_filename(filename)

def read_file_content(filepath):
    """Read content from various file types."""
    ext = get_file_extension(filepath)
    
    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif ext == 'pdf':
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    elif ext in ['doc', 'docx']:
        from docx import Document
        doc = Document(filepath)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def save_text_to_file(text, filepath):
    """Save text content to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

def delete_file(filepath):
    """Safely delete a file if it exists."""
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False 