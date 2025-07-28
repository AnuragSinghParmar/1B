"""
Utility Functions for Adobe Hackathon Round 1B
Contains helper functions for file handling, logging, and validation
"""

import os
import logging
import sys
from typing import Tuple, Dict, Any
from pathlib import Path

def load_persona_job(file_path: str) -> Tuple[str, str]:
    """
    Load persona and job description from file
    
    Args:
        file_path: Path to persona file
        
    Returns:
        Tuple of (persona, job_description)
    """
    # Default values if file doesn't exist
    default_persona = "Research Analyst"
    default_job = "Analyze and summarize key information from research documents"
    
    if not file_path or not os.path.exists(file_path):
        logging.warning(f"Persona file not found: {file_path}, using defaults")
        return default_persona, default_job
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            logging.warning("Persona file is empty, using defaults")
            return default_persona, default_job
        
        lines = content.split('\n')
        
        # First line is persona, rest is job description
        persona = lines[0].strip() if lines else default_persona
        job = ' '.join(lines[1:]).strip() if len(lines) > 1 else default_job
        
        # Ensure non-empty values
        if not persona:
            persona = default_persona
        if not job:
            job = default_job
            
        logging.info(f"Loaded persona: '{persona}' and job: '{job[:100]}...'")
        return persona, job
        
    except Exception as e:
        logging.error(f"Error reading persona file {file_path}: {str(e)}")
        return default_persona, default_job

def setup_logging(level: str = "INFO"):
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Reduce noise from libraries
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def validate_inputs(input_dir: str, output_dir: str):
    """
    Validate input and output directories
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        
    Raises:
        ValueError: If validation fails
    """
    # Check input directory exists
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Check for PDF files in input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise ValueError(f"No PDF files found in input directory: {input_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check output directory is writable
    test_file = os.path.join(output_dir, 'test_write.tmp')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        raise ValueError(f"Output directory is not writable: {output_dir} - {str(e)}")
    
    logging.info(f"Validation passed: {len(pdf_files)} PDFs found in {input_dir}")

def get_memory_usage():
    """
    Get current memory usage in MB
    
    Returns:
        Memory usage in MB or -1 if unable to determine
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return round(memory_mb, 2)
    except ImportError:
        return -1
    except Exception:
        return -1

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return text

def get_pdf_info(pdf_path: str) -> Dict[str, Any]:
    """
    Get basic information about a PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with PDF information
    """
    info = {
        'file_path': pdf_path,
        'file_name': os.path.basename(pdf_path),
        'file_size': 0,
        'file_size_formatted': '0 B',
        'page_count': 0,
        'error': None
    }
    
    try:
        # Get file size
        if os.path.exists(pdf_path):
            info['file_size'] = os.path.getsize(pdf_path)
            info['file_size_formatted'] = format_file_size(info['file_size'])
        
        # Get page count using PyMuPDF
        import fitz
        doc = fitz.open(pdf_path)
        info['page_count'] = len(doc)
        doc.close()
        
    except Exception as e:
        info['error'] = str(e)
        logging.warning(f"Error getting PDF info for {pdf_path}: {str(e)}")
    
    return info

def create_sample_persona_file(file_path: str):
    """
    Create a sample persona file if it doesn't exist
    
    Args:
        file_path: Path where to create the sample file
    """
    if os.path.exists(file_path):
        return
    
    sample_content = """PhD Researcher in Computational Biology
Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks for machine learning applications in drug discovery"""
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        logging.info(f"Created sample persona file: {file_path}")
    except Exception as e:
        logging.error(f"Error creating sample persona file: {str(e)}")

def validate_json_output(json_data: Dict[str, Any]) -> bool:
    """
    Validate that JSON output conforms to expected structure
    
    Args:
        json_data: JSON data to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required top-level keys
        required_keys = ['metadata', 'extracts']
        for key in required_keys:
            if key not in json_data:
                logging.error(f"Missing required key: {key}")
                return False
        
        # Check metadata structure
        metadata = json_data['metadata']
        metadata_keys = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
        for key in metadata_keys:
            if key not in metadata:
                logging.error(f"Missing metadata key: {key}")
                return False
        
        # Check extracts structure
        extracts = json_data['extracts']
        if not isinstance(extracts, list):
            logging.error("Extracts should be a list")
            return False
        
        # Check each extract item
        for i, extract in enumerate(extracts):
            extract_keys = ['document', 'section_title', 'page', 'importance_rank', 'sub_sections']
            for key in extract_keys:
                if key not in extract:
                    logging.error(f"Missing extract key '{key}' in item {i}")
                    return False
        
        logging.info("JSON output validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Error validating JSON output: {str(e)}")
        return False

def print_system_info():
    """Print system information for debugging"""
    import platform
    import sys
    
    logging.info("=== System Information ===")
    logging.info(f"Platform: {platform.platform()}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Architecture: {platform.architecture()}")
    
    # Memory info if available
    memory_mb = get_memory_usage()
    if memory_mb > 0:
        logging.info(f"Current memory usage: {memory_mb} MB")
    
    # Check available libraries
    libraries = ['fitz', 'sklearn', 'sentence_transformers', 'numpy', 'yaml']
    for lib in libraries:
        try:
            __import__(lib)
            logging.info(f"✓ {lib} available")
        except ImportError:
            logging.warning(f"✗ {lib} not available")

def check_constraints(processing_time: float, memory_mb: float) -> Dict[str, bool]:
    """
    Check if processing meets Adobe Hackathon constraints
    
    Args:
        processing_time: Processing time in seconds
        memory_mb: Memory usage in MB
        
    Returns:
        Dictionary with constraint check results
    """
    results = {
        'time_constraint': processing_time <= 60.0,
        'memory_constraint': memory_mb <= 1024.0 if memory_mb > 0 else True,
        'processing_time': processing_time,
        'memory_usage': memory_mb
    }
    
    if results['time_constraint']:
        logging.info(f"✓ Time constraint satisfied: {processing_time:.2f}s <= 60s")
    else:
        logging.warning(f"✗ Time constraint violated: {processing_time:.2f}s > 60s")
    
    if results['memory_constraint']:
        logging.info(f"✓ Memory constraint satisfied: {memory_mb:.2f}MB <= 1024MB")
    else:
        logging.warning(f"✗ Memory constraint violated: {memory_mb:.2f}MB > 1024MB")
    
    return results