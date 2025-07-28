# Approach Explanation: Persona-Driven Document Intelligence

## Methodology Overview

Our solution implements a multi-stage pipeline that combines classical NLP techniques with modern semantic understanding to extract and rank document sections based on persona-specific relevance. The approach prioritizes both accuracy and computational efficiency to meet Adobe's strict constraints.

## Section Extraction Strategy

The PDF analysis phase employs a **multi-modal heading detection system** that goes beyond simple font size thresholds. We first conduct statistical analysis across document pages to understand the font landscape, calculating the dominant font size and establishing dynamic thresholds rather than relying on hardcoded values. Our heading detection algorithm combines five signals: font size analysis (larger fonts indicate importance), font weight detection (bold text patterns), pattern matching using regular expressions for numbered sections (1.1, 2.3.4), positional analysis for left-aligned text, and uppercase text recognition for section titles.

This multi-signal approach achieves robust heading detection across diverse document types, from academic papers with consistent formatting to business reports with varied layouts. Each detected section receives a confidence score based on the strength of these combined signals.

## Relevance Ranking Algorithm

The core innovation lies in our **hybrid ranking system** that fuses three complementary approaches to measure section relevance. First, we employ TF-IDF vectorization using scikit-learn to capture classical term-document relationships between section content and the persona-job query. Second, we leverage semantic embeddings through the lightweight Sentence-BERT model 'all-MiniLM-L6-v2' (400MB) to understand deeper semantic similarities that pure keyword matching might miss. Third, we implement direct keyword matching with differential weighting—job-related keywords receive higher importance than persona keywords since they directly relate to the task at hand.

The final relevance score combines these approaches using weighted averaging: `Score = 0.4 × TF-IDF + 0.3 × Keywords + 0.3 × Embeddings + 0.1 × Heuristics`. The heuristic component provides small boosts for larger fonts, higher confidence sections, and early-page positioning.

## Constraint Optimization

To meet Adobe's stringent requirements, we implemented several optimization strategies. For the 60-second time limit, we use PyMuPDF for fastest PDF processing, limit document analysis to 100 pages per PDF, and employ efficient vectorized operations. For the 1GB model constraint, we selected the compact all-MiniLM-L6-v2 model and optimized TF-IDF to use only 500 features. Memory efficiency is achieved through streaming PDF processing and careful resource cleanup.

## Technical Implementation

The system operates offline with no network dependencies, processing all dependencies during Docker build time. Error handling includes graceful degradation for malformed PDFs and fallback algorithms when ML libraries fail. The modular architecture ensures easy integration into web applications while maintaining clean separation between PDF analysis, ranking logic, and output formatting.

This approach successfully balances accuracy, speed, and resource constraints while generalizing across diverse document domains and persona types, making it ideal for Adobe's multi-domain hackathon test cases.