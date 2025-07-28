#!/usr/bin/env python3
"""
Adobe Hackathon Round 1B - Persona-Driven Document Intelligence
Main execution script that processes PDF collections based on persona and job-to-be-done
"""

import os
import json
import argparse
import datetime
import yaml
import logging
import time
from pathlib import Path

from analyzer import PDFAnalyzer
from ranker import SectionRanker
from utils import load_persona_job, setup_logging, validate_inputs

def main():
    """Main execution function for Round 1B document intelligence system"""
    parser = argparse.ArgumentParser(description='Adobe Round 1B - Persona-Driven Document Intelligence')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input PDFs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output JSON files')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--persona_file', type=str, default='persona.txt', help='File containing persona and job description')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.info("Starting Adobe Round 1B Document Intelligence System")
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Load persona and job description
        persona, job = load_persona_job(args.persona_file)
        logger.info(f"Persona: {persona}")
        logger.info(f"Job-to-be-done: {job}")
        
        # Validate inputs
        validate_inputs(args.input_dir, args.output_dir)
        
        # Get all PDF files from input directory
        pdf_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith('.pdf')])
        if not pdf_files:
            raise ValueError(f"No PDF files found in {args.input_dir}")
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Check constraint: max 10 documents
        if len(pdf_files) > config.get('max_docs', 10):
            logger.warning(f"Found {len(pdf_files)} PDFs, limiting to {config.get('max_docs', 10)}")
            pdf_files = pdf_files[:config.get('max_docs', 10)]
        
        # Extract sections from all PDFs
        all_sections = []
        analyzers = []
        
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(args.input_dir, pdf_file)
            logger.info(f"Processing {pdf_file} ({i+1}/{len(pdf_files)})")
            
            try:
                analyzer = PDFAnalyzer(pdf_path, config)
                sections = analyzer.extract_sections()
                
                # Add document reference to each section
                for section in sections:
                    section['document'] = pdf_file
                
                all_sections.extend(sections)
                analyzers.append((pdf_file, analyzer))
                
                logger.info(f"Extracted {len(sections)} sections from {pdf_file}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                continue
        
        if not all_sections:
            raise ValueError("No sections extracted from any PDF files")
        
        logger.info(f"Total sections extracted: {len(all_sections)}")
        
        # Rank sections based on persona and job relevance
        ranker = SectionRanker(config)
        ranked_sections = ranker.rank_sections(all_sections, persona, job)
        
        logger.info(f"Ranked {len(ranked_sections)} sections by relevance")
        
        # Extract sub-sections for top ranked sections
        for i, section in enumerate(ranked_sections):
            try:
                # Find the corresponding analyzer
                analyzer = next((a for doc, a in analyzers if doc == section['document']), None)
                if analyzer:
                    sub_sections = analyzer.extract_subsections(
                        section.get('start_page', section['page']), 
                        section.get('end_page', section['page'])
                    )
                    section['sub_sections'] = sub_sections
                else:
                    section['sub_sections'] = []
                    
            except Exception as e:
                logger.error(f"Error extracting sub-sections for section {i}: {str(e)}")
                section['sub_sections'] = []
        
        # Prepare output JSON
        output = {
            "metadata": {
                "input_documents": pdf_files,
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.datetime.now().isoformat(),
                "total_sections_analyzed": len(all_sections),
                "processing_time_seconds": round(time.time() - start_time, 2)
            },
            "extracts": []
        }
        
        # Format output according to specification
        for i, section in enumerate(ranked_sections):
            extract = {
                "document": section["document"],
                "section_title": section["title"],
                "page": section["page"],
                "importance_rank": i + 1,
                "relevance_score": round(section.get("score", 0.0), 4),
                "sub_sections": section.get("sub_sections", [])
            }
            output["extracts"].append(extract)
        
        # Write output file
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "challenge1b_output.json")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Processing completed successfully in {total_time:.2f} seconds")
        logger.info(f"✅ Output written to {output_file}")
        logger.info(f"✅ Processed {len(pdf_files)} documents, extracted {len(all_sections)} sections")
        logger.info(f"✅ Ranked and returned top {len(ranked_sections)} relevant sections")
        
        # Verify constraint compliance
        if total_time > 60:
            logger.warning(f"⚠️  Processing time ({total_time:.2f}s) exceeded 60-second constraint")
        else:
            logger.info(f"✅ Processing time constraint satisfied ({total_time:.2f}s < 60s)")
            
        print(f"SUCCESS: Output written to {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        print(f"ERROR: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())