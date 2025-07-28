"""
Section Ranking Module for Adobe Hackathon Round 1B
Ranks document sections based on persona and job-to-be-done relevance
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import re

# Import ML libraries with fallback handling
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using basic ranking")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using TF-IDF only")

class SectionRanker:
    """
    Advanced section ranking system using multiple relevance signals
    Combines semantic similarity, keyword matching, and heuristic scoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize section ranker with configuration
        
        Args:
            config: Configuration dictionary with ranking parameters  
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ranking weights
        self.tfidf_weight = config.get('tfidf_weight', 0.4)
        self.keywords_weight = config.get('keywords_weight', 0.3)
        self.embedding_weight = config.get('embedding_weight', 0.3)
        
        # Parameters
        self.max_features = config.get('max_tfidf_features', 500)
        self.top_k_sections = config.get('top_k_sections', 15)
        
        # Initialize components
        self.tfidf_vectorizer = None
        self.sentence_model = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for ranking"""
        try:
            # Initialize TF-IDF vectorizer
            if SKLEARN_AVAILABLE:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 2),  # Unigrams and bigrams
                    min_df=1,
                    max_df=0.95
                )
                self.logger.info("Initialized TF-IDF vectorizer")
            
            # Initialize sentence transformer (lightweight model)
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                self.sentence_model = SentenceTransformer(model_name)
                self.logger.info(f"Initialized sentence transformer: {model_name}")
                
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            # Fallback to basic keyword matching
    
    def rank_sections(self, sections: List[Dict[str, Any]], persona: str, job: str) -> List[Dict[str, Any]]:
        """
        Rank sections based on relevance to persona and job
        
        Args:
            sections: List of extracted sections
            persona: User persona description
            job: Job-to-be-done description
            
        Returns:
            List of ranked sections with relevance scores
        """
        if not sections:
            self.logger.warning("No sections provided for ranking")
            return []
        
        self.logger.info(f"Ranking {len(sections)} sections for persona: {persona}")
        
        # Combine persona and job as query
        query = f"{persona} {job}"
        
        # Extract text content from sections for analysis
        section_texts = self._extract_section_texts(sections)
        
        # Calculate different relevance scores
        scores = {}
        
        # 1. TF-IDF based similarity
        if self.tfidf_vectorizer and SKLEARN_AVAILABLE:
            scores['tfidf'] = self._calculate_tfidf_scores(section_texts, query)
        else:
            scores['tfidf'] = np.zeros(len(sections))
        
        # 2. Keyword matching scores
        scores['keywords'] = self._calculate_keyword_scores(section_texts, persona, job)
        
        # 3. Semantic embedding similarity
        if self.sentence_model and SENTENCE_TRANSFORMERS_AVAILABLE:
            scores['embedding'] = self._calculate_embedding_scores(section_texts, query)
        else:
            scores['embedding'] = np.zeros(len(sections))
        
        # 4. Heuristic scores based on section characteristics
        scores['heuristic'] = self._calculate_heuristic_scores(sections, persona, job)
        
        # Combine scores using weighted average
        final_scores = self._combine_scores(scores)
        
        # Rank sections by final scores
        ranked_indices = np.argsort(-final_scores)
        
        # Prepare ranked sections with scores
        ranked_sections = []
        for i, idx in enumerate(ranked_indices[:self.top_k_sections]):
            section = sections[idx].copy()
            section['score'] = float(final_scores[idx])
            section['rank'] = i + 1
            
            # Add score breakdown for debugging
            section['score_breakdown'] = {
                'tfidf': float(scores['tfidf'][idx]),
                'keywords': float(scores['keywords'][idx]),
                'embedding': float(scores['embedding'][idx]),
                'heuristic': float(scores['heuristic'][idx])
            }
            
            ranked_sections.append(section)
        
        self.logger.info(f"Ranked sections, top score: {final_scores[ranked_indices[0]]:.4f}")
        return ranked_sections
    
    def _extract_section_texts(self, sections: List[Dict]) -> List[str]:
        """Extract text content from sections for analysis"""
        texts = []
        for section in sections:
            # Combine title and content preview
            text_parts = [section.get('title', '')]
            
            if 'content_preview' in section:
                text_parts.append(section['content_preview'])
            
            combined_text = ' '.join(text_parts).strip()
            texts.append(combined_text)
        
        return texts
    
    def _calculate_tfidf_scores(self, section_texts: List[str], query: str) -> np.ndarray:
        """Calculate TF-IDF based relevance scores"""
        try:
            # Combine all texts with query for fitting
            all_texts = section_texts + [query]
            
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Get query vector (last row) and document vectors
            query_vector = tfidf_matrix[-1]
            doc_vectors = tfidf_matrix[:-1]
            
            # Calculate cosine similarity
            similarities = cosine_similarity(doc_vectors, query_vector).flatten()
            
            self.logger.debug(f"TF-IDF scores calculated, max: {similarities.max():.4f}")
            return similarities
            
        except Exception as e:
            self.logger.error(f"Error calculating TF-IDF scores: {str(e)}")
            return np.zeros(len(section_texts))
    
    def _calculate_keyword_scores(self, section_texts: List[str], persona: str, job: str) -> np.ndarray:
        """Calculate keyword matching scores"""
        # Extract keywords from persona and job
        persona_keywords = self._extract_keywords(persona)
        job_keywords = self._extract_keywords(job)
        all_keywords = persona_keywords + job_keywords
        
        scores = []
        for text in section_texts:
            text_lower = text.lower()
            
            # Count keyword matches with different weights
            persona_matches = sum(1 for kw in persona_keywords if kw in text_lower)
            job_matches = sum(1 for kw in job_keywords if kw in text_lower)
            
            # Weight job keywords higher
            score = persona_matches * 0.4 + job_matches * 0.6
            
            # Normalize by text length to avoid bias toward longer texts
            if text:
                score = score / max(1, np.log(len(text.split()) + 1))
            
            scores.append(score)
        
        scores = np.array(scores)
        
        # Normalize scores to [0, 1] range
        if scores.max() > 0:
            scores = scores / scores.max()
        
        self.logger.debug(f"Keyword scores calculated, max: {scores.max():.4f}")
        return scores
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def _calculate_embedding_scores(self, section_texts: List[str], query: str) -> np.ndarray:
        """Calculate semantic embedding similarity scores"""
        try:
            # Encode query and section texts
            query_embedding = self.sentence_model.encode([query])
            section_embeddings = self.sentence_model.encode(section_texts)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(section_embeddings, query_embedding).flatten()
            
            self.logger.debug(f"Embedding scores calculated, max: {similarities.max():.4f}")
            return similarities
            
        except Exception as e:
            self.logger.error(f"Error calculating embedding scores: {str(e)}")
            return np.zeros(len(section_texts))
    
    def _calculate_heuristic_scores(self, sections: List[Dict], persona: str, job: str) -> np.ndarray:
        """Calculate heuristic scores based on section characteristics"""
        scores = []
        
        for section in sections:
            score = 0.0
            
            # Section title characteristics
            title = section.get('title', '').lower()
            
            # Boost scores for certain section types based on persona
            if 'researcher' in persona.lower() or 'phd' in persona.lower():
                if any(keyword in title for keyword in ['method', 'result', 'conclusion', 'discussion', 'analysis']):
                    score += 0.3
            
            elif 'analyst' in persona.lower() or 'business' in persona.lower():
                if any(keyword in title for keyword in ['revenue', 'financial', 'market', 'trend', 'performance']):
                    score += 0.3
            
            elif 'student' in persona.lower():
                if any(keyword in title for keyword in ['chapter', 'concept', 'example', 'summary', 'key']):
                    score += 0.3
            
            # Font size boost (larger fonts likely more important)
            font_size = section.get('font_size', 12)
            if font_size > 14:
                score += min(0.2, (font_size - 14) * 0.05)
            
            # Confidence boost
            confidence = section.get('confidence', 0)
            score += confidence * 0.2
            
            # Page position (earlier pages slightly preferred)
            page_num = section.get('page', 1)
            if page_num <= 5:
                score += 0.1 * (6 - page_num) / 5
            
            scores.append(score)
        
        scores = np.array(scores)
        
        # Normalize to [0, 1] range
        if scores.max() > 0:
            scores = scores / scores.max()
        
        self.logger.debug(f"Heuristic scores calculated, max: {scores.max():.4f}")
        return scores
    
    def _combine_scores(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine different score types using weighted average"""
        # Normalize weights to sum to 1
        total_weight = self.tfidf_weight + self.keywords_weight + self.embedding_weight
        if total_weight == 0:
            total_weight = 1
        
        tfidf_w = self.tfidf_weight / total_weight
        keywords_w = self.keywords_weight / total_weight
        embedding_w = self.embedding_weight / total_weight
        
        # Combine scores
        final_scores = (
            tfidf_w * scores['tfidf'] +
            keywords_w * scores['keywords'] +
            embedding_w * scores['embedding'] +
            0.1 * scores['heuristic']  # Small heuristic boost
        )
        
        self.logger.debug(f"Combined scores - TF-IDF weight: {tfidf_w:.2f}, Keywords weight: {keywords_w:.2f}, Embedding weight: {embedding_w:.2f}")
        
        return final_scores