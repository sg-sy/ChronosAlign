"""
Wikipedia content extractor for AI API integration.
"""

import requests
import re
from typing import Dict, List, Any, Optional
from urllib.parse import unquote, urlparse
import logging
from bs4 import BeautifulSoup
import json

from .logger import LoggerMixin


class WikipediaExtractor(LoggerMixin):
    """
    Extract and format Wikipedia content for AI API consumption.
    """
    
    def __init__(self):
        """Initialize the Wikipedia extractor."""
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TemporalKnowledgeExtractor/1.0 (research project)'
        })
    
    def extract_wikipedia_content(self, url: str, max_length: int = 8000) -> Dict[str, Any]:
        """
        Extract content from a Wikipedia page and format it for AI APIs.
        
        Args:
            url: Wikipedia page URL
            max_length: Maximum content length to return
            
        Returns:
            Dictionary containing formatted content and metadata
        """
        try:
            # Clean and validate URL
            clean_url = self._clean_wikipedia_url(url)
            if not clean_url:
                return self._create_error_response("Invalid Wikipedia URL")
            
            # Fetch the page
            response = self.session.get(clean_url, timeout=30)
            response.raise_for_status()
            
            # Parse the content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract page information
            page_info = self._extract_page_info(soup, clean_url)
            
            # Extract tables BEFORE removing them from DOM
            tables = []
            for table in soup.find_all('table', class_='wikitable'):
                try:
                    # Extract headers - try multiple approaches
                    headers = []
                    
                    # Method 1: Look for th elements in the first row
                    first_row = table.find('tr')
                    if first_row:
                        for th in first_row.find_all('th'):
                            header_text = th.get_text(strip=True)
                            if header_text:
                                headers.append(header_text)
                    
                    # Method 2: If no headers found, try td elements in first row
                    if not headers and first_row:
                        for td in first_row.find_all('td'):
                            header_text = td.get_text(strip=True)
                            if header_text:
                                headers.append(header_text)
                    
                    # Method 3: If still no headers, create generic headers
                    if not headers:
                        # Count columns in first data row
                        first_data_row = table.find('tr')
                        if first_data_row:
                            col_count = len(first_data_row.find_all(['td', 'th']))
                            headers = [f"Column {i+1}" for i in range(col_count)]
                    
                    # Extract rows
                    rows = []
                    table_rows = table.find_all('tr')
                    
                    # Skip the first row if it was used for headers
                    start_row = 1 if headers else 0
                    
                    for tr in table_rows[start_row:]:
                        row = []
                        for cell in tr.find_all(['td', 'th']):
                            cell_text = cell.get_text(strip=True)
                            # Clean up the text (remove extra whitespace, newlines)
                            cell_text = ' '.join(cell_text.split())
                            row.append(cell_text)
                        
                        # Only add non-empty rows
                        if row and any(cell for cell in row):
                            rows.append(row)
                    
                    # Only add tables that have meaningful content
                    if headers and rows and len(rows) > 0:
                        # Ensure all rows have the same number of columns as headers
                        max_cols = len(headers)
                        cleaned_rows = []
                        
                        for row in rows:
                            # Pad or truncate row to match header count
                            while len(row) < max_cols:
                                row.append('')
                            if len(row) > max_cols:
                                row = row[:max_cols]
                            cleaned_rows.append(row)
                        
                        tables.append({
                            'headers': headers,
                            'rows': cleaned_rows
                        })
                        
                except Exception as e:
                    self.log_warning(f"Error extracting table: {e}")
                    continue
            
            # Add tables to page info
            if tables:
                page_info['tables'] = tables
            
            # Extract main content (this will remove tables from DOM)
            content = self._extract_main_content(soup)
            
            # Format for AI APIs
            formatted_content = self._format_for_ai_apis(content, page_info, max_length)
            
            return {
                'success': True,
                'url': clean_url,
                'title': page_info.get('title', ''),
                'content': formatted_content,
                'metadata': page_info,
                'content_length': len(formatted_content)
            }
            
        except requests.exceptions.RequestException as e:
            self.log_error(f"Failed to fetch Wikipedia page: {e}")
            return self._create_error_response(f"Failed to fetch page: {str(e)}")
        except Exception as e:
            self.log_error(f"Error extracting Wikipedia content: {e}")
            return self._create_error_response(f"Extraction error: {str(e)}")
    
    def _clean_wikipedia_url(self, url: str) -> Optional[str]:
        """
        Clean and validate a Wikipedia URL.
        
        Args:
            url: Raw URL string
            
        Returns:
            Cleaned URL or None if invalid
        """
        try:
            # Handle incomplete URLs
            if url.endswith('%'):
                url = url[:-1]  # Remove trailing percent sign
            
            # Ensure it's a Wikipedia URL
            if 'wikipedia.org' not in url:
                return None
            
            # Parse and reconstruct URL
            parsed = urlparse(url)
            if not parsed.scheme:
                url = 'https://' + url
            
            # Ensure it's an English Wikipedia page
            if '/wiki/' not in url:
                return None
            
            return url
            
        except Exception as e:
            self.log_error(f"Error cleaning URL: {e}")
            return None
    
    def _extract_page_info(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract metadata from the Wikipedia page.
        
        Args:
            soup: BeautifulSoup object
            url: Page URL
            
        Returns:
            Dictionary with page metadata
        """
        info = {
            'url': url,
            'title': '',
            'summary': '',
            'categories': [],
            'last_modified': '',
            'word_count': 0
        }
        
        try:
            # Extract title
            title_elem = soup.find('h1', {'id': 'firstHeading'})
            if title_elem:
                info['title'] = title_elem.get_text().strip()
            
            # Extract summary (first paragraph)
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if content_div:
                first_para = content_div.find('p', recursive=False)
                if first_para:
                    info['summary'] = first_para.get_text().strip()
            
            # Extract categories
            category_links = soup.find_all('a', href=re.compile(r'/wiki/Category:'))
            info['categories'] = [link.get_text().strip() for link in category_links[:10]]
            
            # Extract last modified date
            last_modified = soup.find('div', {'id': 'footer-info-lastmod'})
            if last_modified:
                info['last_modified'] = last_modified.get_text().strip()
            
        except Exception as e:
            self.log_warning(f"Error extracting page info: {e}")
        
        return info
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract the main content from the Wikipedia page.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Cleaned content text
        """
        content = []
        
        try:
            # Find the main content area
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return ""
            
            # Remove unwanted elements
            for unwanted in content_div.find_all(['script', 'style', 'table', 'nav', 'sup']):
                unwanted.decompose()
            
            # Extract paragraphs
            paragraphs = content_div.find_all('p')
            for p in paragraphs:
                text = p.get_text().strip()
                if text and len(text) > 50:  # Filter out short paragraphs
                    content.append(text)
            
            # Extract section headers
            headers = content_div.find_all(['h2', 'h3', 'h4'])
            for header in headers:
                text = header.get_text().strip()
                if text:
                    content.append(f"\n## {text}\n")
            
        except Exception as e:
            self.log_warning(f"Error extracting main content: {e}")
        
        return '\n\n'.join(content)
    
    def _format_for_ai_apis(self, content: str, page_info: Dict[str, Any], max_length: int) -> str:
        """
        Format content for AI API consumption.
        
        Args:
            content: Raw content text
            page_info: Page metadata
            max_length: Maximum content length
            
        Returns:
            Formatted content for AI APIs
        """
        # Create structured format
        formatted_parts = []
        
        # Add title
        if page_info.get('title'):
            formatted_parts.append(f"Title: {page_info['title']}")
        
        # Add summary
        if page_info.get('summary'):
            formatted_parts.append(f"Summary: {page_info['summary']}")
        
        # Add main content
        if content:
            # Truncate if necessary
            if len(content) > max_length:
                content = content[:max_length] + "... [Content truncated]"
            formatted_parts.append(f"Content:\n{content}")
        
        # Add tables if available
        if page_info.get('tables'):
            formatted_parts.append("Tables:")
            for i, table in enumerate(page_info['tables'], 1):
                table_text = f"\nTable {i}:\n"
                
                # Add headers
                if table.get('headers'):
                    table_text += "| " + " | ".join(table['headers']) + " |\n"
                    table_text += "|" + "|".join(["---"] * len(table['headers'])) + "|\n"
                
                # Add rows
                if table.get('rows'):
                    for row in table['rows']:
                        table_text += "| " + " | ".join(row) + " |\n"
                
                formatted_parts.append(table_text)
        
        # Add metadata
        if page_info.get('categories'):
            categories = ', '.join(page_info['categories'][:5])
            formatted_parts.append(f"Categories: {categories}")
        
        if page_info.get('last_modified'):
            formatted_parts.append(f"Last Modified: {page_info['last_modified']}")
        
        return '\n\n'.join(formatted_parts)
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error_message: Error description
            
        Returns:
            Error response dictionary
        """
        return {
            'success': False,
            'error': error_message,
            'content': '',
            'metadata': {},
            'content_length': 0
        }
    
    def extract_for_anthropic(self, url: str, max_length: int = 8000) -> str:
        """
        Extract content formatted specifically for Anthropic Claude API.
        
        Args:
            url: Wikipedia page URL
            max_length: Maximum content length
            
        Returns:
            Formatted content string for Anthropic
        """
        result = self.extract_wikipedia_content(url, max_length)
        
        if not result['success']:
            return f"Error: {result['error']}"
        
        # Get the formatted content (which now includes tables)
        formatted_content = result['content']
        
        # Format for Anthropic's preferred style
        content = f"""Wikipedia Article: {result['title']}

{formatted_content}

Source: {result['url']}"""
        
        return content
    
    def extract_for_openai(self, url: str, max_length: int = 8000) -> str:
        """
        Extract content formatted specifically for OpenAI API.
        
        Args:
            url: Wikipedia page URL
            max_length: Maximum content length
            
        Returns:
            Formatted content string for OpenAI
        """
        result = self.extract_wikipedia_content(url, max_length)
        
        if not result['success']:
            return f"Error: {result['error']}"
        
        # Get the formatted content (which now includes tables)
        formatted_content = result['content']
        
        # Format for OpenAI's preferred style
        content = f"""Wikipedia Page: {result['title']}

{formatted_content}

Reference: {result['url']}"""
        
        return content
    
    def extract_structured_for_ai(self, url: str, max_length: int = 8000) -> Dict[str, Any]:
        """
        Extract content in a structured format suitable for AI APIs.
        
        Args:
            url: Wikipedia page URL
            max_length: Maximum content length
            
        Returns:
            Structured dictionary for AI consumption
        """
        result = self.extract_wikipedia_content(url, max_length)
        
        if not result['success']:
            return {
                'error': result['error'],
                'content': '',
                'metadata': {}
            }
        
        # Return structured format
        return {
            'title': result['title'],
            'content': result['content'],
            'metadata': result['metadata'],
            'url': result['url'],
            'content_length': result['content_length']
        }


def extract_wikipedia_content(url: str, api_type: str = "general", max_length: int = 8000) -> str:
    """
    Convenience function to extract Wikipedia content for AI APIs.
    
    Args:
        url: Wikipedia page URL
        api_type: Target API type ("anthropic", "openai", or "general")
        max_length: Maximum content length
        
    Returns:
        Formatted content string
    """
    extractor = WikipediaExtractor()
    
    if api_type.lower() == "anthropic":
        return extractor.extract_for_anthropic(url, max_length)
    elif api_type.lower() == "openai":
        return extractor.extract_for_openai(url, max_length)
    else:
        result = extractor.extract_wikipedia_content(url, max_length)
        return result['content'] if result['success'] else f"Error: {result['error']}"


if __name__ == "__main__":
    # Example usage
    extractor = WikipediaExtractor()
    
    # Test with a Wikipedia URL
    test_url = "https://en.wikipedia.org/wiki/Visitors"
    result = extractor.extract_wikipedia_content(test_url)
    
    print(f"Success: {result['success']}")
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"Content length: {result.get('content_length', 0)}")
    print(f"Content preview: {result.get('content', '')[:200]}...") 