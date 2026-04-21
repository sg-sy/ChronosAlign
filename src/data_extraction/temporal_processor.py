"""
Temporal data processing utilities.
"""

import re
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple, Union
from dateutil import parser
import pytz

from ..utils.logger import LoggerMixin


class TemporalProcessor(LoggerMixin):
    """
    Process temporal data from Wikidata.
    """
    
    def __init__(self):
        """Initialize the temporal processor."""
        super().__init__()
        
        # Wikidata time precision levels
        self.precision_levels = {
            0: "billion years",
            1: "hundred million years", 
            2: "ten million years",
            3: "million years",
            4: "hundred thousand years",
            5: "ten thousand years",
            6: "millennium",
            7: "century",
            8: "decade",
            9: "year",
            10: "month",
            11: "day",
            12: "hour",
            13: "minute",
            14: "second"
        }
    
    def parse_wikidata_time(self, time_value: str, precision: Optional[int] = None) -> Dict[str, Any]:
        """
        Parse a Wikidata time value.
        
        Args:
            time_value: Wikidata time value (e.g., "+1950-01-01T00:00:00Z/9")
            precision: Time precision level
            
        Returns:
            Dictionary with parsed temporal information
        """
        try:
            # Extract the date part (before the precision indicator)
            if '/' in time_value:
                date_part = time_value.split('/')[0]
            else:
                date_part = time_value
            
            # Remove the '+' prefix if present
            if date_part.startswith('+'):
                date_part = date_part[1:]
            
            # Parse the date
            parsed_date = parser.parse(date_part)
            
            # Determine precision if not provided
            if precision is None:
                precision = self._infer_precision(time_value)
            
            return {
                'original_value': time_value,
                'parsed_date': parsed_date,
                'year': parsed_date.year,
                'month': parsed_date.month if precision >= 10 else None,
                'day': parsed_date.day if precision >= 11 else None,
                'precision': precision,
                'precision_label': self.precision_levels.get(precision, "unknown"),
                'is_approximate': self._is_approximate(time_value),
                'timezone': str(parsed_date.tzinfo) if parsed_date.tzinfo else None
            }
            
        except Exception as e:
            self.log_warning(f"Failed to parse time value '{time_value}': {e}")
            return {
                'original_value': time_value,
                'parsed_date': None,
                'precision': precision,
                'precision_label': self.precision_levels.get(precision, "unknown") if precision else "unknown",
                'is_approximate': self._is_approximate(time_value),
                'error': str(e)
            }
    
    def _infer_precision(self, time_value: str) -> int:
        """
        Infer the precision level from a time value.
        
        Args:
            time_value: Wikidata time value
            
        Returns:
            Precision level
        """
        # Check for precision indicator
        if '/' in time_value:
            precision_part = time_value.split('/')[1]
            try:
                return int(precision_part)
            except ValueError:
                pass
        
        # Infer from date format
        if 'T' in time_value:  # Has time component
            return 11  # day precision
        elif len(time_value.split('-')) == 3:  # Has day
            return 11  # day precision
        elif len(time_value.split('-')) == 2:  # Has month
            return 10  # month precision
        else:  # Year only
            return 9  # year precision
    
    def _is_approximate(self, time_value: str) -> bool:
        """
        Check if a time value is approximate.
        
        Args:
            time_value: Wikidata time value
            
        Returns:
            True if approximate, False otherwise
        """
        return '~' in time_value or 'circa' in time_value.lower()
    
    def calculate_duration(self, start_time: Dict[str, Any], end_time: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Calculate duration between two temporal points.
        
        Args:
            start_time: Start time dictionary
            end_time: End time dictionary
            
        Returns:
            Duration information or None if calculation fails
        """
        try:
            if not start_time.get('parsed_date') or not end_time.get('parsed_date'):
                return None
            
            start_date = start_time['parsed_date']
            end_date = end_time['parsed_date']
            
            if start_date > end_date:
                self.log_warning("Start date is after end date")
                return None
            
            duration = end_date - start_date
            
            return {
                'start_time': start_time,
                'end_time': end_time,
                'duration_days': duration.days,
                'duration_years': duration.days / 365.25,
                'duration_months': duration.days / 30.44,
                'is_exact': not (start_time.get('is_approximate') or end_time.get('is_approximate'))
            }
            
        except Exception as e:
            self.log_error(f"Failed to calculate duration: {e}")
            return None
    
    def compare_times(self, time1: Dict[str, Any], time2: Dict[str, Any]) -> Optional[str]:
        """
        Compare two temporal points.
        
        Args:
            time1: First time dictionary
            time2: Second time dictionary
            
        Returns:
            Comparison result: 'before', 'after', 'same', or None if comparison fails
        """
        try:
            if not time1.get('parsed_date') or not time2.get('parsed_date'):
                return None
            
            date1 = time1['parsed_date']
            date2 = time2['parsed_date']
            
            if date1 < date2:
                return 'before'
            elif date1 > date2:
                return 'after'
            else:
                return 'same'
                
        except Exception as e:
            self.log_error(f"Failed to compare times: {e}")
            return None
    
    def validate_temporal_consistency(self, temporal_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate temporal consistency of a set of temporal data.
        
        Args:
            temporal_data: List of temporal data dictionaries
            
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        for i, data in enumerate(temporal_data):
            # Check for parsing errors
            if 'error' in data:
                issues.append(f"Parsing error in item {i}: {data['error']}")
            
            # Check for very old dates (before 1000 CE)
            if data.get('year') and data['year'] < 1000:
                warnings.append(f"Very old date in item {i}: {data['year']}")
            
            # Check for future dates
            if data.get('year') and data['year'] > 2024:
                warnings.append(f"Future date in item {i}: {data['year']}")
        
        # Check for logical inconsistencies in pairs
        for i in range(0, len(temporal_data) - 1, 2):
            if i + 1 < len(temporal_data):
                start = temporal_data[i]
                end = temporal_data[i + 1]
                
                if start.get('parsed_date') and end.get('parsed_date'):
                    if start['parsed_date'] > end['parsed_date']:
                        issues.append(f"Start date after end date in pair {i//2}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_items': len(temporal_data)
        }
    
    def format_temporal_answer(self, temporal_data: Dict[str, Any], 
                             reasoning_type: str) -> str:
        """
        Format temporal data into a natural language answer.
        
        Args:
            temporal_data: Temporal data dictionary
            reasoning_type: Type of temporal reasoning
            
        Returns:
            Formatted answer string
        """
        if reasoning_type == "point_in_time":
            return self._format_point_in_time(temporal_data)
        elif reasoning_type == "duration":
            return self._format_duration(temporal_data)
        elif reasoning_type == "before_after":
            return self._format_before_after(temporal_data)
        elif reasoning_type == "sequence":
            return self._format_sequence(temporal_data)
        elif reasoning_type == "comparison":
            return self._format_comparison(temporal_data)
        else:
            return self._format_generic(temporal_data)
    
    def _format_point_in_time(self, data: Dict[str, Any]) -> str:
        """Format a point-in-time answer."""
        if not data.get('parsed_date'):
            return "The exact date is not available."
        
        year = data['year']
        precision = data.get('precision', 9)
        
        if precision >= 11:  # Day precision
            return f"{data['parsed_date'].strftime('%B %d, %Y')}"
        elif precision >= 10:  # Month precision
            return f"{data['parsed_date'].strftime('%B %Y')}"
        else:  # Year precision
            return f"{year}"
    
    def _format_duration(self, data: Dict[str, Any]) -> str:
        """Format a duration answer."""
        if 'duration_days' not in data:
            return "Duration information is not available."
        
        days = data['duration_days']
        years = data['duration_years']
        
        if years >= 1:
            return f"Approximately {years:.1f} years"
        elif days >= 30:
            months = days / 30.44
            return f"Approximately {months:.1f} months"
        else:
            return f"{days} days"
    
    def _format_before_after(self, data: Dict[str, Any]) -> str:
        """Format a before/after comparison answer."""
        comparison = data.get('comparison')
        if comparison == 'before':
            return "Yes, the first event happened before the second."
        elif comparison == 'after':
            return "No, the first event happened after the second."
        elif comparison == 'same':
            return "They happened at the same time."
        else:
            return "The temporal relationship cannot be determined."
    
    def _format_sequence(self, data: Dict[str, Any]) -> str:
        """Format a sequence answer."""
        events = data.get('events', [])
        if len(events) < 2:
            return "Insufficient information to determine sequence."
        
        sorted_events = sorted(events, key=lambda x: x.get('parsed_date', datetime.max))
        return f"First: {sorted_events[0].get('label', 'Unknown')}, Second: {sorted_events[1].get('label', 'Unknown')}"
    
    def _format_comparison(self, data: Dict[str, Any]) -> str:
        """Format a comparison answer."""
        if 'duration_days' not in data:
            return "Duration information is not available for comparison."
        
        return f"The first lasted {data['duration_days']} days, the second lasted {data.get('duration_days_2', 'unknown')} days."
    
    def _format_generic(self, data: Dict[str, Any]) -> str:
        """Format a generic temporal answer."""
        if data.get('parsed_date'):
            return f"The date is {data['parsed_date'].strftime('%Y-%m-%d')}"
        else:
            return "Temporal information is not available." 