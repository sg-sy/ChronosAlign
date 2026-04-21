"""
Main Wikidata extractor for temporal knowledge.
"""

import click
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from .sparql_client import WikidataSPARQLClient
from .temporal_processor import TemporalProcessor
from ..utils.logger import setup_logger, LoggerMixin
from ..utils.file_utils import load_yaml, save_data, ensure_directory


class WikidataExtractor(LoggerMixin):
    """
    Main extractor for temporal knowledge from Wikidata.
    """
    
    def __init__(self, config_path: str = "config/extraction_config.yaml"):
        """
        Initialize the Wikidata extractor.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__()
        
        # Load configuration
        self.config = load_yaml(config_path)
        
        # Get timeout configuration
        wikidata_config = self.config.get('wikidata', {})
        timeout_config = wikidata_config.get('timeout', 120)
        
        # Initialize components
        self.sparql_client = WikidataSPARQLClient(
            endpoint=wikidata_config.get('endpoint', "https://query.wikidata.org/sparql"),
            user_agent=wikidata_config.get('user_agent', "TemporalKnowledgeExtractor/1.0"),
            rate_limit_delay=self.config['extraction']['rate_limit_delay'],
            timeout=timeout_config
        )
        
        self.temporal_processor = TemporalProcessor()
        
        # Setup logging
        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            'WikidataExtractor',
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
    
    def extract_temporal_entities(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract entities with temporal properties from Wikidata.
        
        Args:
            limit: Maximum number of entities to extract
            
        Returns:
            List of temporal entities
        """
        if limit is None:
            limit = self.config['extraction']['max_entities']
        
        self.log_info(f"Starting extraction of up to {limit} temporal entities...")
        
        # Load SPARQL queries
        queries_config = load_yaml("config/sparql_queries.yaml")
        
        # Execute the main temporal entities query
        query = queries_config['temporal_entities'].format(limit=limit)
        bindings = self.sparql_client.get_bindings(query)
        
        entities = []
        for binding in tqdm(bindings, desc="Processing entities"):
            entity = self._process_entity_binding(binding)
            if entity:
                entities.append(entity)
        
        self.log_info(f"Extracted {len(entities)} temporal entities")
        return entities
    
    def _process_entity_binding(self, binding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single entity binding from SPARQL results.
        
        Args:
            binding: SPARQL binding dictionary
            
        Returns:
            Processed entity dictionary or None if invalid
        """
        try:
            entity_id = binding.get('entity', {}).get('value', '').split('/')[-1]
            entity_label = binding.get('entityLabel', {}).get('value', '')
            entity_type = binding.get('entityType', {}).get('value', '').split('/')[-1]
            entity_type_label = binding.get('entityTypeLabel', {}).get('value', '')
            
            if not entity_id or not entity_label:
                return None
            
            # Get temporal properties for this entity
            temporal_properties = self.sparql_client.get_temporal_properties(entity_id)
            
            # Process temporal properties
            processed_properties = []
            for prop in temporal_properties:
                processed_prop = self._process_temporal_property(prop)
                if processed_prop:
                    processed_properties.append(processed_prop)
            
            return {
                'entity_id': entity_id,
                'entity_label': entity_label,
                'entity_type': entity_type,
                'entity_type_label': entity_type_label,
                'temporal_properties': processed_properties,
                'extraction_timestamp': time.time()
            }
            
        except Exception as e:
            self.log_error(f"Error processing entity binding: {e}")
            return None
    
    def _process_temporal_property(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a temporal property from SPARQL results.
        
        Args:
            prop: Temporal property binding
            
        Returns:
            Processed temporal property or None if invalid
        """
        try:
            property_id = prop.get('property', {}).get('value', '').split('/')[-1]
            property_label = prop.get('propertyLabel', {}).get('value', '')
            time_value = prop.get('value', {}).get('value', '')
            precision = prop.get('precision', {}).get('value')
            
            if not property_id or not time_value:
                return None
            
            # Parse the temporal value
            parsed_time = self.temporal_processor.parse_wikidata_time(time_value, precision)
            
            return {
                'property_id': property_id,
                'property_label': property_label,
                'time_value': time_value,
                'parsed_time': parsed_time
            }
            
        except Exception as e:
            self.log_error(f"Error processing temporal property: {e}")
            return None
    
    def extract_birth_death_entities(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract entities with both birth and death dates.
        
        Args:
            limit: Maximum number of entities to extract
            
        Returns:
            List of birth-death entities
        """
        if limit is None:
            limit = self.config['extraction']['max_entities']
        
        self.log_info(f"Extracting {limit} entities with birth and death dates...")
        
        queries_config = load_yaml("config/sparql_queries.yaml")
        query = queries_config['birth_death_entities'].format(limit=limit)
        bindings = self.sparql_client.get_bindings(query)
        
        entities = []
        for binding in tqdm(bindings, desc="Processing birth-death entities"):
            entity = self._process_birth_death_binding(binding)
            if entity:
                entities.append(entity)
        
        return entities
    
    def _process_birth_death_binding(self, binding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a birth-death entity binding.
        
        Args:
            binding: SPARQL binding dictionary
            
        Returns:
            Processed entity dictionary or None if invalid
        """
        try:
            entity_id = binding.get('entity', {}).get('value', '').split('/')[-1]
            entity_label = binding.get('entityLabel', {}).get('value', '')
            birth_date = binding.get('birthDate', {}).get('value', '')
            death_date = binding.get('deathDate', {}).get('value', '')
            birth_precision = binding.get('birthPrecision', {}).get('value')
            death_precision = binding.get('deathPrecision', {}).get('value')
            
            if not entity_id or not entity_label or not birth_date or not death_date:
                return None
            
            # Parse temporal values
            parsed_birth = self.temporal_processor.parse_wikidata_time(birth_date, birth_precision)
            parsed_death = self.temporal_processor.parse_wikidata_time(death_date, death_precision)
            
            # Calculate duration
            duration = self.temporal_processor.calculate_duration(parsed_birth, parsed_death)
            
            return {
                'entity_id': entity_id,
                'entity_label': entity_label,
                'birth_date': birth_date,
                'death_date': death_date,
                'parsed_birth': parsed_birth,
                'parsed_death': parsed_death,
                'duration': duration,
                'extraction_timestamp': time.time()
            }
            
        except Exception as e:
            self.log_error(f"Error processing birth-death binding: {e}")
            return None
    
    def extract_comparison_entities(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract entities for temporal comparisons.
        
        Args:
            limit: Maximum number of entity pairs to extract
            
        Returns:
            List of comparison entity pairs
        """
        if limit is None:
            limit = self.config['extraction']['max_entities']
        
        self.log_info(f"Extracting {limit} entity pairs for temporal comparisons...")
        
        queries_config = load_yaml("config/sparql_queries.yaml")
        query = queries_config['comparison_entities'].format(limit=limit)
        bindings = self.sparql_client.get_bindings(query)
        
        comparisons = []
        for binding in tqdm(bindings, desc="Processing comparison entities"):
            comparison = self._process_comparison_binding(binding)
            if comparison:
                comparisons.append(comparison)
        
        return comparisons
    
    def _process_comparison_binding(self, binding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a comparison entity binding.
        
        Args:
            binding: SPARQL binding dictionary
            
        Returns:
            Processed comparison dictionary or None if invalid
        """
        try:
            entity1_id = binding.get('entity1', {}).get('value', '').split('/')[-1]
            entity1_label = binding.get('entity1Label', {}).get('value', '')
            entity2_id = binding.get('entity2', {}).get('value', '').split('/')[-1]
            entity2_label = binding.get('entity2Label', {}).get('value', '')
            time1 = binding.get('time1', {}).get('value', '')
            time2 = binding.get('time2', {}).get('value', '')
            
            if not entity1_id or not entity2_id or not time1 or not time2:
                return None
            
            # Parse temporal values
            parsed_time1 = self.temporal_processor.parse_wikidata_time(time1)
            parsed_time2 = self.temporal_processor.parse_wikidata_time(time2)
            
            # Compare times
            comparison = self.temporal_processor.compare_times(parsed_time1, parsed_time2)
            
            return {
                'entity1_id': entity1_id,
                'entity1_label': entity1_label,
                'entity2_id': entity2_id,
                'entity2_label': entity2_label,
                'time1': time1,
                'time2': time2,
                'parsed_time1': parsed_time1,
                'parsed_time2': parsed_time2,
                'comparison': comparison,
                'extraction_timestamp': time.time()
            }
            
        except Exception as e:
            self.log_error(f"Error processing comparison binding: {e}")
            return None
    
    def save_extracted_data(self, data: List[Dict[str, Any]], 
                           filename: str, data_type: str = "raw"):
        """
        Save extracted data to file.
        
        Args:
            data: Data to save
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
        
        save_data(data, file_path, file_format)
        self.log_info(f"Saved {len(data)} items to {file_path}")
    
    def run_extraction(self, output_prefix: str = "temporal_entities"):
        """
        Run the complete extraction process.
        
        Args:
            output_prefix: Prefix for output files
        """
        self.log_info("Starting Wikidata temporal knowledge extraction...")
        
        # Test connection
        if not self.sparql_client.test_connection():
            self.log_error("Failed to connect to Wikidata")
            return
        
        # Extract different types of temporal entities
        temporal_entities = self.extract_temporal_entities()
        self.save_extracted_data(temporal_entities, f"{output_prefix}_general.jsonl")
        
        birth_death_entities = self.extract_birth_death_entities()
        self.save_extracted_data(birth_death_entities, f"{output_prefix}_birth_death.jsonl")
        
        comparison_entities = self.extract_comparison_entities()
        self.save_extracted_data(comparison_entities, f"{output_prefix}_comparisons.jsonl")
        
        self.log_info("Extraction completed successfully!")


@click.command()
@click.option('--config', default='config/extraction_config.yaml', 
              help='Path to configuration file')
@click.option('--output-prefix', default='temporal_entities', 
              help='Prefix for output files')
def main(config: str, output_prefix: str):
    """
    Extract temporal knowledge from Wikidata.
    """
    extractor = WikidataExtractor(config)
    extractor.run_extraction(output_prefix)


if __name__ == "__main__":
    main() 