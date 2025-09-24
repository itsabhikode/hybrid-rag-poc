"""
JSON serialization utilities for handling Neo4j DateTime objects and other non-serializable types.
"""

from datetime import datetime, date, time
from typing import Any, Dict
from neo4j.time import DateTime as Neo4jDateTime, Date as Neo4jDate, Time as Neo4jTime


def serialize_neo4j_object(obj: Any) -> Any:
    """
    Recursively serialize Neo4j objects and other non-JSON-serializable types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    elif isinstance(obj, (Neo4jDateTime, datetime)):
        return obj.isoformat()
    elif isinstance(obj, (Neo4jDate, date)):
        return obj.isoformat()
    elif isinstance(obj, (Neo4jTime, time)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_neo4j_object(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_neo4j_object(item) for item in obj]
    elif isinstance(obj, set):
        return [serialize_neo4j_object(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting to dict
        return serialize_neo4j_object(obj.__dict__)
    else:
        # For primitive types (str, int, float, bool), return as-is
        return obj


def safe_json_response(data: Any) -> Dict[str, Any]:
    """
    Create a safe JSON response by serializing all non-serializable objects.
    
    Args:
        data: Data to serialize
        
    Returns:
        JSON-serializable dictionary
    """
    return serialize_neo4j_object(data)


def format_indicator_response(indicator_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format indicator data for API response, handling DateTime serialization.
    
    Args:
        indicator_data: Raw indicator data from Neo4j
        
    Returns:
        Formatted indicator data
    """
    return {
        "value": indicator_data.get("value", ""),
        "normalized_value": indicator_data.get("normalized_value", ""),
        "indicator_type": indicator_data.get("indicator_type", ""),
        "confidence_score": indicator_data.get("confidence_score", 0),
        "created_at": serialize_neo4j_object(indicator_data.get("created_at", "")),
        "metadata": {
            "threat_level": indicator_data.get("threat_level", ""),
            "is_verified": indicator_data.get("is_verified", False),
            "extraction_method": indicator_data.get("extraction_method", "")
        }
    }


def format_document_response(document_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format document data for API response, handling DateTime serialization.
    
    Args:
        document_data: Raw document data from Neo4j
        
    Returns:
        Formatted document data
    """
    return {
        "filename": document_data.get("filename", ""),
        "file_path": document_data.get("file_path", ""),
        "created_at": serialize_neo4j_object(document_data.get("created_at", "")),
        "file_size": document_data.get("file_size", 0),
        "page_count": document_data.get("page_count", 0)
    }


def format_relationship_response(relationship_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format relationship data for API response, handling DateTime serialization.
    
    Args:
        relationship_data: Raw relationship data from Neo4j
        
    Returns:
        Formatted relationship data
    """
    return {
        "relationship_type": relationship_data.get("relationship_type", ""),
        "indicator": format_indicator_response(relationship_data.get("indicator", {})),
        "related_entity": serialize_neo4j_object(relationship_data.get("related_entity", {})),
        "relationship_properties": serialize_neo4j_object(relationship_data.get("relationship", {}))
    }
