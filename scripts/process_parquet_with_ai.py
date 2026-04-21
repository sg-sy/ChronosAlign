#!/usr/bin/env python3
"""
Script to process parquet files containing Wikipedia content through the AI question generator.
This script takes a parquet file with the same properties as WikipediaContent.parquet and
runs each row through the AI question generator to create temporal datasets.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd

# Add the repository root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.ai_question_generator import process_parquet_file
from prompt_config.config import system_prompt, user_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Process parquet files through AI question generator"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input parquet file"
    )
    parser.add_argument(
        "--output-file",
        "-o",
        help="Path to the output file (JSON or CSV). If not specified, will use input filename with _qa suffix"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "huggingface"],
        default="anthropic",
        help="AI provider to use (default: anthropic)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for the specified provider (optional, will use environment variable if not provided)"
    )
    parser.add_argument(
        "--model",
        help="Model name to use (optional, will use default for the provider)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run Hugging Face models on (default: cpu)"
    )
    parser.add_argument(
        "--cache-dir",
        help="Cache directory for Hugging Face models (optional)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        help="Number of questions to generate per content item (default: unlimited)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Maximum tokens for the response (default: 4000)"
    )
    parser.add_argument(
        "--content-column",
        default="content",
        help="Name of the column containing the content to process (default: content)"
    )
    parser.add_argument(
        "--success-column",
        default="success",
        help="Name of the column indicating successful extraction (default: success)"
    )
    parser.add_argument(
        "--min-content-length",
        type=int,
        default=100,
        help="Minimum content length to process (default: 100)"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "parquet"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--custom-system-prompt",
        help="Custom system prompt (optional, will use default from config if not provided)"
    )
    parser.add_argument(
        "--custom-user-prompt",
        help="Custom user prompt template (optional, will use default from config if not provided)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    # Determine output file path
    if args.output_file:
        output_path = args.output_file
    else:
        input_path = Path(args.input_file)
        output_path = input_path.parent / f"{input_path.stem}_qa.{args.output_format}"
    
    # Use custom prompts if provided, otherwise use defaults
    system_prompt_to_use = args.custom_system_prompt or system_prompt
    user_prompt_to_use = args.custom_user_prompt or user_prompt
    
    print(f"Processing parquet file: {args.input_file}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")
    print(f"Output file: {output_path}")
    print(f"Output format: {args.output_format}")
    print(f"Number of questions per content: {args.num_questions or 'unlimited'}")
    print(f"Min content length: {args.min_content_length}")
    print("-" * 50)
    
    try:
        # Process the parquet file
        qa_pairs = process_parquet_file(
            parquet_file_path=args.input_file,
            system_prompt=system_prompt_to_use,
            user_prompt=user_prompt_to_use,
            provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            device=args.device,
            cache_dir=args.cache_dir,
            num_questions_per_content=args.num_questions,
            max_tokens=args.max_tokens,
            content_column=args.content_column,
            success_column=args.success_column,
            min_content_length=args.min_content_length
        )
        
        if not qa_pairs:
            print("No QA pairs were generated. Check the input file and parameters.")
            sys.exit(1)
        
        print(f"\nGenerated {len(qa_pairs)} QA pairs")
        
        # Save the results
        if args.output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        elif args.output_format == "csv":
            df = pd.DataFrame(qa_pairs)
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif args.output_format == "parquet":
            df = pd.DataFrame(qa_pairs)
            df.to_parquet(output_path, index=False)
        
        print(f"Results saved to: {output_path}")
        
        # Print some statistics
        print("\nStatistics:")
        print(f"Total QA pairs: {len(qa_pairs)}")
        
        # Count unique source rows
        source_rows = set()
        for qa_pair in qa_pairs:
            if 'source_row_index' in qa_pair:
                source_rows.add(qa_pair['source_row_index'])
        
        print(f"Source rows processed: {len(source_rows)}")
        
        # Show sample of generated QA pairs
        print("\nSample QA pairs:")
        for i, qa_pair in enumerate(qa_pairs[:3]):
            print(f"\n{i+1}. Question: {qa_pair.get('question', 'N/A')}")
            print(f"   Answer: {qa_pair.get('answer', 'N/A')}")
            if 'event_year' in qa_pair:
                print(f"   Event Year: {qa_pair.get('event_year', 'N/A')}")
            if 'subject' in qa_pair:
                print(f"   Subject: {qa_pair.get('subject', 'N/A')}")
        
        if len(qa_pairs) > 3:
            print(f"\n... and {len(qa_pairs) - 3} more QA pairs")
        
    except Exception as e:
        print(f"Error processing parquet file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 