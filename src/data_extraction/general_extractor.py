"""
General SPARQL extractor for Wikidata queries.
"""

import click
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

from .sparql_client import WikidataSPARQLClient
from ..utils.logger import setup_logger, LoggerMixin
from ..utils.file_utils import load_yaml, save_data, ensure_directory


class GeneralSPARQLExtractor(LoggerMixin):
    """
    General extractor for executing SPARQL queries against Wikidata.
    """
    
    def __init__(self, config_path: str = "config/extraction_config.yaml"):
        """
        Initialize the general SPARQL extractor.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__()
        
        # Load configuration
        self.config = load_yaml(config_path)
        
        # Get timeout configuration
        wikidata_config = self.config.get('wikidata', {})
        timeout_config = wikidata_config.get('timeout', 120)
        
        # Initialize SPARQL client
        self.sparql_client = WikidataSPARQLClient(
            endpoint=wikidata_config.get('endpoint', "https://query.wikidata.org/sparql"),
            user_agent=wikidata_config.get('user_agent', "TemporalKnowledgeExtractor/1.0"),
            rate_limit_delay=self.config['extraction']['rate_limit_delay'],
            timeout=timeout_config
        )
        
        # Setup logging
        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            'GeneralSPARQLExtractor',
            log_file=log_config.get('file'),
            level=log_config.get('level', 'INFO')
        )
        
        # Ensure output directories exist
        for dir_path in [
            self.config['output']['raw_data_dir'],
            self.config['output']['processed_data_dir'],
            self.config['output']['final_data_dir']
        ]:
            ensure_directory(dir_path)
    
    def execute_query(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query and return results.
        
        Args:
            query: SPARQL query string
            limit: Optional limit to add to query
            
        Returns:
            List of query results as dictionaries
        """
        if limit:
            # Add LIMIT clause if not already present
            if 'LIMIT' not in query.upper():
                query = query.rstrip() + f"\nLIMIT {limit}"
        
        self.log_info(f"Executing SPARQL query with limit: {limit}")
        
        try:
            bindings = self.sparql_client.get_bindings(query)
            results = []
            
            for binding in tqdm(bindings, desc="Processing query results"):
                processed_binding = self._process_binding(binding)
                if processed_binding:
                    results.append(processed_binding)
            
            self.log_info(f"Query returned {len(results)} results")
            return results
            
        except Exception as e:
            self.log_error(f"Error executing query: {e}")
            return []
    
    def _process_binding(self, binding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single SPARQL binding into a clean dictionary.
        
        Args:
            binding: SPARQL binding dictionary
            
        Returns:
            Processed binding dictionary or None if invalid
        """
        try:
            processed = {}
            
            for var_name, var_data in binding.items():
                if isinstance(var_data, dict) and 'value' in var_data:
                    value = var_data['value']
                    
                    # Extract entity ID from Wikidata URIs
                    if value.startswith('http://www.wikidata.org/entity/'):
                        value = value.split('/')[-1]
                    elif value.startswith('http://www.wikidata.org/'):
                        value = value.split('/')[-1]
                    
                    processed[var_name] = value
                else:
                    processed[var_name] = var_data
            
            return processed
            
        except Exception as e:
            self.log_error(f"Error processing binding: {e}")
            return None
    
    def execute_query_from_file(self, query_file: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query from a file.
        
        Args:
            query_file: Path to file containing SPARQL query
            limit: Optional limit to add to query
            
        Returns:
            List of query results as dictionaries
        """
        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                query = f.read().strip()
            
            self.log_info(f"Loaded query from {query_file}")
            return self.execute_query(query, limit)
            
        except Exception as e:
            self.log_error(f"Error reading query file {query_file}: {e}")
            return []
    
    def save_results(self, results: List[Dict[str, Any]], 
                    filename: str, data_type: str = "raw"):
        """
        Save query results to file.
        
        Args:
            results: Query results to save
            filename: Output filename
            data_type: Type of data (raw, processed, final)
        """
        if data_type == "raw":
            output_dir = self.config['output']['raw_data_dir']
        elif data_type == "processed":
            output_dir = self.config['output']['processed_data_dir']
        else:
            output_dir = self.config['output']['final_data_dir']
        
        file_path = Path(output_dir) / filename
        file_format = self.config['output']['file_format']
        
        save_data(results, file_path, file_format)
        self.log_info(f"Saved {len(results)} results to {file_path}")
    
    def extract_with_query(self, query: Union[str, Path], 
                          output_filename: str,
                          limit: Optional[int] = None,
                          data_type: str = "raw") -> List[Dict[str, Any]]:
        """
        Extract data using a SPARQL query and save results.
        
        Args:
            query: SPARQL query string or path to query file
            output_filename: Name for output file
            limit: Optional limit for query results
            data_type: Type of data for output directory
            
        Returns:
            List of extracted results
        """
        self.log_info(f"Starting extraction with query: {query}")
        
        # Test connection
        if not self.sparql_client.test_connection():
            self.log_error("Failed to connect to Wikidata")
            return []
        
        # Execute query
        if isinstance(query, (str, Path)) and Path(query).exists():
            results = self.execute_query_from_file(str(query), limit)
        else:
            results = self.execute_query(str(query), limit)
        
        # Save results
        if results:
            self.save_results(results, output_filename, data_type)
        
        return results
    
    def batch_extract(self, query_files: List[str], 
                     output_prefix: str = "extracted",
                     limit: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute multiple queries from files in batch.
        
        Args:
            query_files: List of paths to query files
            output_prefix: Prefix for output files
            limit: Optional limit for all queries
            
        Returns:
            Dictionary mapping query files to their results
        """
        self.log_info(f"Starting batch extraction of {len(query_files)} queries")
        
        results = {}
        
        for query_file in query_files:
            try:
                # Generate output filename based on query file name
                query_name = Path(query_file).stem
                output_filename = f"{output_prefix}_{query_name}.jsonl"
                
                # Execute query
                query_results = self.extract_with_query(
                    query_file, 
                    output_filename, 
                    limit
                )
                
                results[query_file] = query_results
                
            except Exception as e:
                self.log_error(f"Error processing query file {query_file}: {e}")
                results[query_file] = []
        
        self.log_info(f"Batch extraction completed. Processed {len(query_files)} queries")
        return results


@click.command()
@click.option('--query', '-q', required=True, 
              help='SPARQL query string or path to query file')
@click.option('--output', '-o', default='extracted_results.jsonl',
              help='Output filename')
@click.option('--limit', '-l', type=int, 
              help='Limit number of results')
@click.option('--config', default='config/extraction_config.yaml',
              help='Path to configuration file')
@click.option('--data-type', default='raw',
              help='Data type for output directory (raw, processed, final)')
def main(query: str, output: str, limit: Optional[int], config: str, data_type: str):
    """
    Execute a SPARQL query and save results.
    """
    extractor = GeneralSPARQLExtractor(config)
    results = extractor.extract_with_query(query, output, limit, data_type)
    
    if results:
        print(f"Successfully extracted {len(results)} results to {output}")
    else:
        print("No results extracted")


@click.command()
@click.option('--query-dir', '-d', default='config',
              help='Directory containing query files')
@click.option('--output-prefix', '-p', default='extracted',
              help='Prefix for output files')
@click.option('--limit', '-l', type=int, 
              help='Limit number of results per query')
@click.option('--config', default='config/extraction_config.yaml',
              help='Path to configuration file')
@click.option('--pattern', default='*.txt',
              help='File pattern to match query files')
def batch_main(query_dir: str, output_prefix: str, limit: Optional[int], 
               config: str, pattern: str):
    """
    Execute multiple SPARQL queries from files in batch.
    """
    extractor = GeneralSPARQLExtractor(config)
    
    # Find query files
    query_dir_path = Path(query_dir)
    query_files = list(query_dir_path.glob(pattern))
    
    if not query_files:
        print(f"No query files found in {query_dir} matching pattern {pattern}")
        return
    
    print(f"Found {len(query_files)} query files")
    
    # Execute batch extraction
    results = extractor.batch_extract(
        [str(f) for f in query_files],
        output_prefix,
        limit
    )
    
    # Print summary
    total_results = sum(len(r) for r in results.values())
    print(f"Batch extraction completed. Total results: {total_results}")


if __name__ == "__main__":
    # You can use either main() or batch_main() depending on your needs
    main() 