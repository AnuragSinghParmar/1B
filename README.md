# Adobe Hackathon Round 1B: Persona-Driven Document Intelligence

## Overview

This solution implements an advanced **Persona-Driven Document Intelligence System** for Adobe's "Connecting the Dots" Challenge Round 1B. The system analyzes collections of PDF documents and extracts the most relevant sections based on a specific user persona and their job-to-be-done.

## Key Features

- **Multi-Modal PDF Analysis**: Combines font analysis, pattern recognition, and layout understanding
- **Advanced Ranking System**: Uses TF-IDF, semantic embeddings, and heuristic scoring
- **Lightweight & Fast**: Processes 3-10 PDFs in under 60 seconds using CPU-only
- **Offline Operation**: No internet access required during execution
- **Constraint Compliant**: Stays within 1GB model size and 60-second time limits

## Architecture

The solution consists of four main components:

1. **PDFAnalyzer** (`analyzer.py`): Extracts structured sections from PDF documents
2. **SectionRanker** (`ranker.py`): Ranks sections based on persona and job relevance
3. **Main Controller** (`main.py`): Orchestrates the entire processing pipeline
4. **Utilities** (`utils.py`): Helper functions for file handling and validation

## Quick Start

### 1. Build Docker Image

```bash
docker build --platform linux/amd64 -t adobe-round1b .
```

### 2. Prepare Input

Create the following directory structure:
```
data/
├── input/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── document3.pdf
└── output/
```

Create a `persona.txt` file with your persona and job description:
```
Investment Analyst
Analyze revenue trends, R&D investments, and market positioning strategies
```

### 3. Run the Solution

```bash
docker run --rm \
  -v $(pwd)/data/input:/app/input \
  -v $(pwd)/data/output:/app/output \
  -v $(pwd)/persona.txt:/app/persona.txt \
  --network none \
  adobe-round1b
```

### 4. View Results

Check `data/output/challenge1b_output.json` for the ranked section extracts.

## Input Specification

### Document Collection
- **Format**: PDF files (3-10 documents)
- **Content**: Any domain (research papers, textbooks, financial reports, etc.)
- **Size**: Up to 100 pages per document

### Persona Definition
- **Format**: Text file (`persona.txt`)
- **Structure**: First line = persona, remaining lines = job description
- **Examples**: 
  - PhD Researcher, Business Analyst, Undergraduate Student

### Job-to-be-Done
- **Purpose**: Concrete task the persona needs to accomplish
- **Examples**:
  - "Comprehensive literature review focusing on methodologies"
  - "Analyze revenue trends and market positioning"
  - "Identify key concepts for exam preparation"

## Output Format

The system generates a JSON file with the following structure:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "Investment Analyst",
    "job_to_be_done": "Analyze revenue trends...",
    "processing_timestamp": "2025-01-15T10:30:00",
    "processing_time_seconds": 25.4
  },
  "extracts": [
    {
      "document": "doc1.pdf",
      "section_title": "Revenue Analysis",
      "page": 5,
      "importance_rank": 1,
      "relevance_score": 0.8947,
      "sub_sections": [
        {
          "page": 5,
          "summary": "Revenue increased by 15% year-over-year..."
        }
      ]
    }
  ]
}
```

## Technical Implementation

### PDF Section Extraction
- **Font Analysis**: Statistical analysis of font sizes and weights
- **Pattern Recognition**: Regular expressions for numbered sections and headings
- **Layout Understanding**: Position-based heuristics for heading detection
- **Multi-modal Fusion**: Combines multiple signals for robust extraction

### Relevance Ranking Algorithm
1. **TF-IDF Similarity**: Classical text similarity using scikit-learn
2. **Semantic Embeddings**: Dense vector representations using Sentence-BERT
3. **Keyword Matching**: Direct keyword overlap with persona/job
4. **Heuristic Scoring**: Font size, position, and confidence-based boosts

### Scoring Formula
```
Final Score = 0.4 × TF-IDF + 0.3 × Keywords + 0.3 × Embeddings + 0.1 × Heuristics
```

## Performance Characteristics

| Metric | Requirement | Achieved |
|--------|-------------|----------|
| Processing Time | ≤ 60 seconds | ~20-40 seconds |
| Model Size | ≤ 1GB | ~400MB |
| Memory Usage | CPU only | <500MB RAM |
| Platform | linux/amd64 | ✓ Compatible |
| Network Access | None | ✓ Offline |

## Dependencies

Core libraries used:
- `pymupdf`: PDF processing and text extraction
- `scikit-learn`: TF-IDF vectorization and similarity
- `sentence-transformers`: Semantic embeddings (all-MiniLM-L6-v2)
- `numpy`: Numerical computations
- `PyYAML`: Configuration management

## Configuration

Adjust parameters in `config.yaml`:
- **Ranking weights**: Modify TF-IDF, keyword, and embedding contributions
- **Analysis thresholds**: Font size and section length minimums
- **Model selection**: Choose different Sentence-BERT models
- **Performance limits**: Processing time and memory constraints

## Test Cases Supported

The solution handles diverse scenarios:

### Academic Research
- **Documents**: Research papers on specialized topics
- **Personas**: PhD Researchers, Post-docs, Academics
- **Jobs**: Literature reviews, methodology analysis, benchmarking

### Business Analysis
- **Documents**: Annual reports, financial statements, market research
- **Personas**: Investment Analysts, Business Consultants
- **Jobs**: Revenue analysis, competitive positioning, trend identification

### Educational Content
- **Documents**: Textbooks, course materials, reference guides
- **Personas**: Students (undergraduate, graduate)
- **Jobs**: Exam preparation, concept understanding, study guides

## Error Handling

The system includes robust error handling:
- **Malformed PDFs**: Graceful degradation with logging
- **Missing dependencies**: Fallback to basic algorithms
- **Memory constraints**: Automatic page limiting
- **Processing timeouts**: Early termination with partial results

## Logging & Debugging

Comprehensive logging system:
- **Performance metrics**: Processing time, memory usage
- **Score breakdowns**: Individual component contributions
- **Error tracking**: Detailed error messages and stack traces
- **Progress monitoring**: Real-time processing updates

## Constraints Compliance

✅ **Processing Time**: Consistently under 60 seconds for 3-10 documents  
✅ **Model Size**: Uses lightweight models totaling <1GB  
✅ **CPU Only**: No GPU dependencies, runs on standard hardware  
✅ **Offline**: All processing done without internet access  
✅ **Format**: Output matches Adobe's exact JSON specification  

## Development Notes

### Code Organization
- Modular design with clear separation of concerns
- Type hints and comprehensive documentation
- Error handling at all levels
- Configuration-driven behavior

### Testing
- Handles diverse document types and layouts
- Tested with various persona/job combinations
- Performance validated under constraint limits
- Edge case handling verified

### Future Enhancements
- Support for additional document formats
- Fine-tuned domain-specific models
- Advanced layout analysis
- Multi-language support

## Troubleshooting

### Common Issues

**No output generated**:
- Check input directory contains PDF files
- Verify persona.txt exists and is formatted correctly
- Check Docker volume mounts are correct

**Processing too slow**:
- Reduce number of input documents
- Adjust `max_pages_per_doc` in config.yaml
- Increase Docker memory allocation

**Low relevance scores**:
- Refine persona and job descriptions
- Adjust ranking weights in config.yaml
- Check document content matches persona domain

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure input format matches specification
4. Review configuration parameters

---

**Built for Adobe Hackathon 2025 - "Connecting the Dots" Challenge**  
**Designed to be fast, accurate, and easily integrable into web applications**