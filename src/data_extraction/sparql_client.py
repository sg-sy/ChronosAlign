"""
SPARQL client for querying Wikidata.
"""

import requests
import time
from typing import Dict, List, Any, Optional, Union
from urllib.parse import quote
import logging

from ..utils.logger import LoggerMixin


class WikidataSPARQLClient(LoggerMixin):
    """
    Client for querying Wikidata using SPARQL.
    """
    
    def __init__(self, endpoint: str = "https://query.wikidata.org/sparql", 
                 user_agent: str = "TemporalKnowledgeExtractor/1.0", 
                 rate_limit_delay: float = 1.0,
                 timeout: Union[int, tuple] = 120):
        """
        Initialize the SPARQL client.
        
        Args:
            endpoint: Wikidata SPARQL endpoint
            user_agent: User agent string
            rate_limit_delay: Delay between requests in seconds
            timeout: Request timeout in seconds (can be int or tuple for (connect, read))
        """
        super().__init__()
        self.endpoint = endpoint
        self.user_agent = user_agent
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'application/sparql-results+json'
        })
    
    def query(self, sparql_query: str, timeout: Union[int, tuple] = None) -> Optional[Dict[str, Any]]:
        """
        Execute a SPARQL query.
        
        Args:
            sparql_query: SPARQL query string
            timeout: Request timeout in seconds (overrides default if provided)
            
        Returns:
            Query results as dictionary or None if failed
        """
        try:
            # Add rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Prepare the request
            params = {
                'query': sparql_query,
                'format': 'json'
            }
            
            # Use provided timeout or default
            request_timeout = timeout if timeout is not None else self.timeout
            
            self.log_debug(f"Executing SPARQL query with timeout {request_timeout}s: {sparql_query[:100]}...")
            
            response = self.session.get(
                self.endpoint,
                params=params,
                timeout=request_timeout
            )
            
            response.raise_for_status()
            
            result = response.json()
            self.log_debug(f"Query returned {len(result.get('results', {}).get('bindings', []))} results")
            
            return result
            
        except requests.exceptions.Timeout as e:
            self.log_error(f"SPARQL query timed out after {request_timeout}s: {e}")
            return None
        except requests.exceptions.RequestException as e:
            self.log_error(f"SPARQL query failed: {e}")
            return None
        except Exception as e:
            self.log_error(f"Unexpected error during SPARQL query: {e}")
            return None
    
    def get_bindings(self, sparql_query: str, timeout: Union[int, tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query and return the bindings.
        
        Args:
            sparql_query: SPARQL query string
            timeout: Request timeout in seconds (overrides default if provided)
            
        Returns:
            List of binding dictionaries
        """
        result = self.query(sparql_query, timeout=timeout)
        if result and 'results' in result:
            return result['results']['bindings']
        return []
    
    def format_query(self, query_template: str, **kwargs) -> str:
        """
        Format a query template with parameters.
        
        Args:
            query_template: Query template string
            **kwargs: Parameters to substitute
            
        Returns:
            Formatted query string
        """
        return query_template.format(**kwargs)
    
    def get_entity_info(self, entity_id: str, timeout: Union[int, tuple] = None) -> Optional[Dict[str, Any]]:
        """
        Get basic information about an entity.
        
        Args:
            entity_id: Wikidata entity ID (e.g., "Q42")
            timeout: Request timeout in seconds (overrides default if provided)
            
        Returns:
            Entity information dictionary
        """
        query = f"""
        SELECT ?entity ?entityLabel ?entityDescription ?entityType ?entityTypeLabel
        WHERE {{
            BIND(wd:{entity_id} AS ?entity)
            OPTIONAL {{ ?entity schema:description ?entityDescription . FILTER(LANG(?entityDescription) = "en") }}
            OPTIONAL {{ ?entity wdt:P31 ?entityType }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }}
        }}
        """
        
        bindings = self.get_bindings(query, timeout=timeout)
        if bindings:
            return bindings[0]
        return None
    
    def get_temporal_properties(self, entity_id: str, timeout: Union[int, tuple] = None) -> List[Dict[str, Any]]:
        """
        Get temporal properties for an entity.
        
        Args:
            entity_id: Wikidata entity ID
            timeout: Request timeout in seconds (overrides default if provided)
            
        Returns:
            List of temporal properties
        """
        query = f"""
        SELECT ?property ?propertyLabel ?value ?valueLabel ?precision
        WHERE {{
            BIND(wd:{entity_id} AS ?entity)
            
            VALUES ?property {{ wdt:P569 wdt:P570 wdt:P580 wdt:P582 wdt:P585 wdt:P1319 wdt:P1326 wdt:P2031 wdt:P2032 wdt:P571 wdt:P576 wdt:P1191 wdt:P1619 wdt:P1629 }}
            
            ?entity ?property ?value .
            
            OPTIONAL {{
                ?value wikibase:timePrecision ?precision .
            }}
            
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }}
        }}
        """
        
        return self.get_bindings(query, timeout=timeout)
    
    def search_entities(self, search_term: str, entity_type: str = None, 
                       limit: int = 10, timeout: Union[int, tuple] = None) -> List[Dict[str, Any]]:
        """
        Search for entities by label.
        
        Args:
            search_term: Search term
            entity_type: Optional entity type filter
            limit: Maximum number of results
            timeout: Request timeout in seconds (overrides default if provided)
            
        Returns:
            List of matching entities
        """
        query = f"""
        SELECT ?entity ?entityLabel ?entityDescription ?entityType ?entityTypeLabel
        WHERE {{
            ?entity rdfs:label ?entityLabel .
            FILTER(CONTAINS(LCASE(?entityLabel), "{search_term.lower()}"))
            FILTER(LANG(?entityLabel) = "en")
            
            OPTIONAL {{ ?entity schema:description ?entityDescription . FILTER(LANG(?entityDescription) = "en") }}
            OPTIONAL {{ ?entity wdt:P31 ?entityType }}
            
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }}
        }}
        """
        
        if entity_type:
            query += f'FILTER(?entityType = wd:{entity_type})'
        
        query += f"LIMIT {limit}"
        
        return self.get_bindings(query, timeout=timeout)
    
    def test_connection(self, timeout: Union[int, tuple] = None) -> bool:
        """
        Test the connection to Wikidata.
        
        Args:
            timeout: Request timeout in seconds (overrides default if provided)
            
        Returns:
            True if connection is successful, False otherwise
        """
        test_query = "SELECT ?count WHERE { BIND(1 AS ?count) }"
        result = self.query(test_query, timeout=timeout)
        return result is not None 