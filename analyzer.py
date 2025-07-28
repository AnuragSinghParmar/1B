"""
PDF Analysis Module for Adobe Hackathon Round 1B
Handles PDF processing, section extraction, and content analysis
"""

import fitz  # PyMuPDF
import re
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter

class PDFAnalyzer:
    """
    Advanced PDF analyzer that extracts structured sections and content
    Uses multi-modal analysis combining font, layout, and pattern recognition
    """
    
    def __init__(self, pdf_path: str, config: Dict[str, Any]):
        """
        Initialize PDF analyzer with configuration
        
        Args:
            pdf_path: Path to PDF file
            config: Configuration dictionary with analysis parameters
        """
        self.pdf_path = pdf_path
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            self.doc = fitz.open(pdf_path)
            self.logger.info(f"Opened PDF: {pdf_path} ({len(self.doc)} pages)")
        except Exception as e:
            self.logger.error(f"Failed to open PDF {pdf_path}: {str(e)}")
            raise
        
        # Analysis thresholds
        self.font_size_threshold = config.get('font_size_threshold', 12)
        self.min_section_length = config.get('min_section_length', 10)
        self.max_pages_per_doc = config.get('max_pages_per_doc', 100)
        
        # Heading patterns for detection
        self.heading_patterns = [
            r'^(\d+\.?\s+)',                    # Numbered: "1. ", "1.1 ", "2.3.4 "
            r'^([A-Z][A-Z\s]{3,})\s*$',         # ALL CAPS: "INTRODUCTION"
            r'^([A-Z][a-z]+(\s+[A-Z][a-z]+)*):',# Title Case with colon: "Chapter One:"
            r'^(Chapter|Section|Part)\s+\d+',   # Explicit chapters/sections
            r'^([A-Z][a-z]+\s+\d+)',           # "Chapter 1", "Section 2"
        ]
    
    def extract_sections(self) -> List[Dict[str, Any]]:
        """
        Extract sections from PDF using multi-modal analysis
        
        Returns:
            List of section dictionaries with title, page, and metadata
        """
        self.logger.info("Starting section extraction")
        sections = []
        
        # Limit pages to avoid memory issues
        max_pages = min(len(self.doc), self.max_pages_per_doc)
        
        # First pass: collect font statistics
        font_stats = self._analyze_font_statistics(max_pages)
        
        # Second pass: extract sections using font and pattern analysis
        for page_num in range(max_pages):
            try:
                page = self.doc[page_num]
                page_sections = self._extract_page_sections(page, page_num + 1, font_stats)
                sections.extend(page_sections)
                
            except Exception as e:
                self.logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                continue
        
        # Post-process sections: clean, deduplicate, and enrich
        processed_sections = self._process_sections(sections)
        
        self.logger.info(f"Extracted {len(processed_sections)} sections")
        return processed_sections
    
    def _analyze_font_statistics(self, max_pages: int) -> Dict[str, Any]:
        """Analyze font patterns across the document for threshold calculation"""
        font_sizes = []
        font_families = []
        
        for page_num in range(min(max_pages, 10)):  # Sample first 10 pages
            try:
                page = self.doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if len(text) > 3:  # Only consider meaningful text
                                font_sizes.append(span["size"])
                                font_families.append(span["font"])
                                
            except Exception as e:
                self.logger.warning(f"Error analyzing page {page_num + 1}: {str(e)}")
                continue
        
        if not font_sizes:
            return {"main_font_size": 12, "heading_threshold": 14}
        
        # Calculate font statistics
        font_counter = Counter([round(size) for size in font_sizes])
        main_font_size = font_counter.most_common(1)[0][0]
        
        # Set heading threshold based on document characteristics
        heading_threshold = max(main_font_size + 2, self.font_size_threshold)
        
        return {
            "main_font_size": main_font_size,
            "heading_threshold": heading_threshold,
            "font_sizes": font_sizes,
            "common_fonts": Counter(font_families).most_common(5)
        }
    
    def _extract_page_sections(self, page, page_num: int, font_stats: Dict) -> List[Dict[str, Any]]:
        """Extract sections from a single page"""
        sections = []
        
        try:
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        
                        if len(text) < self.min_section_length:
                            continue
                        
                        # Multi-modal heading detection
                        if self._is_section_heading(span, text, font_stats):
                            section = {
                                "title": self._clean_section_title(text),
                                "page": page_num,
                                "start_page": page_num,
                                "end_page": page_num,
                                "font_size": span["size"],
                                "font_family": span["font"],
                                "bbox": span["bbox"],
                                "confidence": self._calculate_confidence(span, text, font_stats)
                            }
                            sections.append(section)
                            
        except Exception as e:
            self.logger.warning(f"Error extracting sections from page {page_num}: {str(e)}")
        
        return sections
    
    def _is_section_heading(self, span: Dict, text: str, font_stats: Dict) -> bool:
        """
        Determine if text span is a section heading using multi-modal analysis
        
        Args:
            span: Text span with font information
            text: Text content
            font_stats: Document font statistics
            
        Returns:
            Boolean indicating if text is likely a heading
        """
        # Font size analysis
        font_size = span["size"]
        is_large_font = font_size >= font_stats["heading_threshold"]
        
        # Font weight analysis
        font_name = span["font"].lower()
        is_bold = any(weight in font_name for weight in ["bold", "black", "heavy", "demi"])
        
        # Pattern matching
        matches_pattern = any(re.match(pattern, text) for pattern in self.heading_patterns)
        
        # Length heuristics
        is_reasonable_length = 5 <= len(text) <= 200
        
        # Position analysis (approximate)
        bbox = span.get("bbox", [0, 0, 0, 0])
        is_left_aligned = bbox[0] < 100  # Simple left alignment check
        
        # All caps check (but not too long)
        is_all_caps = text.isupper() and 5 <= len(text) <= 50
        
        # Scoring system
        score = 0
        if is_large_font: score += 3
        if is_bold: score += 2
        if matches_pattern: score += 3
        if is_all_caps: score += 2
        if is_left_aligned: score += 1
        
        # Require minimum score and reasonable length
        return score >= 3 and is_reasonable_length
    
    def _calculate_confidence(self, span: Dict, text: str, font_stats: Dict) -> float:
        """Calculate confidence score for section heading detection"""
        confidence = 0.0
        
        # Font size confidence
        font_size = span["size"]
        if font_size >= font_stats["heading_threshold"] + 2:
            confidence += 0.3
        elif font_size >= font_stats["heading_threshold"]:
            confidence += 0.2
        
        # Pattern matching confidence
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                confidence += 0.3
                break
        
        # Font weight confidence
        if any(weight in span["font"].lower() for weight in ["bold", "black"]):
            confidence += 0.2
        
        # Text characteristics
        if text.isupper() and 5 <= len(text) <= 50:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _clean_section_title(self, text: str) -> str:
        """Clean and normalize section titles"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove trailing punctuation except periods
        text = re.sub(r'[^\w\s\.]+$', '', text)
        
        # Limit length
        if len(text) > 200:
            text = text[:200] + "..."
        
        return text
    
    def _process_sections(self, sections: List[Dict]) -> List[Dict]:
        """Post-process sections: deduplicate, merge, and enrich"""
        if not sections:
            return []
        
        # Sort by page and position
        sections.sort(key=lambda x: (x["page"], x.get("bbox", [0])[1]))
        
        # Remove duplicates based on similar titles
        unique_sections = []
        seen_titles = set()
        
        for section in sections:
            title_key = self._normalize_title_for_comparison(section["title"])
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_sections.append(section)
        
        # Enrich sections with additional content
        enriched_sections = []
        for section in unique_sections:
            try:
                content = self._extract_section_content(section)
                section["content_preview"] = content[:500] if content else ""
                enriched_sections.append(section)
            except Exception as e:
                self.logger.warning(f"Error enriching section '{section['title']}': {str(e)}")
                enriched_sections.append(section)
        
        return enriched_sections
    
    def _normalize_title_for_comparison(self, title: str) -> str:
        """Normalize title for duplicate detection"""
        # Convert to lowercase, remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _extract_section_content(self, section: Dict) -> str:
        """Extract content following a section heading"""
        try:
            page_num = section["page"] - 1
            page = self.doc[page_num]
            
            # Get full page text for content extraction
            full_text = page.get_text()
            
            # Find section title in text and extract following content
            title = section["title"]
            title_index = full_text.find(title)
            
            if title_index != -1:
                # Extract text after the title (next 1000 chars)
                start_idx = title_index + len(title)
                content = full_text[start_idx:start_idx + 1000]
                return content.strip()
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Error extracting content for section: {str(e)}")
            return ""
    
    def extract_subsections(self, start_page: int, end_page: int) -> List[Dict[str, Any]]:
        """
        Extract sub-sections and summaries for a page range
        
        Args:
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed)
            
        Returns:
            List of sub-section dictionaries with page and summary
        """
        sub_sections = []
        
        for page_num in range(start_page - 1, min(end_page, len(self.doc))):
            try:
                page = self.doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    # Create summary by taking first few sentences
                    sentences = re.split(r'[.!?]+', text)
                    summary_sentences = []
                    char_count = 0
                    
                    for sentence in sentences[:10]:  # Max 10 sentences
                        sentence = sentence.strip()
                        if sentence and char_count + len(sentence) < 300:
                            summary_sentences.append(sentence)
                            char_count += len(sentence)
                        else:
                            break
                    
                    summary = '. '.join(summary_sentences)
                    if summary and not summary.endswith('.'):
                        summary += '.'
                    
                    sub_section = {
                        "page": page_num + 1,
                        "summary": summary,
                        "word_count": len(text.split()),
                        "char_count": len(text)
                    }
                    sub_sections.append(sub_section)
                    
            except Exception as e:
                self.logger.warning(f"Error extracting sub-section from page {page_num + 1}: {str(e)}")
                continue
        
        self.logger.debug(f"Extracted {len(sub_sections)} sub-sections from pages {start_page}-{end_page}")
        return sub_sections
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'doc') and self.doc:
            try:
                self.doc.close()
            except:
                pass