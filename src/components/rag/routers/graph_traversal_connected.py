import os
import sys
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from shared.services.rag_service import rag_agent
from shared.utils.json_serializer import safe_json_response, format_relationship_response

router = APIRouter()


@router.get("/relationships/{indicator_value}")
async def graph_traversal_connected(
    indicator_value: str,
    depth: int = Query(2, description="Traversal depth (1-3)", ge=1, le=3),
    relationship_types: Optional[str] = Query(None, description="Comma-separated list of relationship types to filter")
):
    """
    Graph traversal for connected entities.
    Returns all relationships and connected entities for a specific indicator.
    """
    try:
        if not indicator_value.strip():
            raise HTTPException(status_code=400, detail="Indicator value is required")
        
        # Get relationships using RAG agent
        relationships = rag_agent.get_relationships(indicator_value)
        
        # Filter by relationship types if specified
        if relationship_types:
            filter_types = [t.strip().lower() for t in relationship_types.split(",")]
            relationships = [
                rel for rel in relationships 
                if rel.get("relationship_type", "").lower() in filter_types
            ]
        
        # Format the response with proper DateTime serialization
        formatted_relationships = []
        connected_entities = set()
        relationship_stats = {}
        
        for rel in relationships:
            relationship_type = rel.get("relationship_type", "unknown")
            relationship_stats[relationship_type] = relationship_stats.get(relationship_type, 0) + 1
            
            # Format relationship data using the serializer
            formatted_rel = format_relationship_response(rel)
            formatted_relationships.append(formatted_rel)
            
            # Track connected entities
            related_entity = rel.get("related_entity", {})
            if related_entity:
                entity_id = related_entity.get("id", "")
                if entity_id:
                    connected_entities.add(entity_id)
        
        # Analyze the network
        analysis = {
            "total_relationships": len(formatted_relationships),
            "unique_connected_entities": len(connected_entities),
            "relationship_type_distribution": relationship_stats,
            "most_common_relationship": max(relationship_stats.items(), key=lambda x: x[1])[0] if relationship_stats else None,
            "network_density": len(formatted_relationships) / max(len(connected_entities), 1),
            "traversal_depth": depth
        }
        
        response_data = {
            "indicator": indicator_value,
            "relationships": formatted_relationships,
            "analysis": analysis,
            "filters": {
                "depth": depth,
                "relationship_types": relationship_types
            },
            "status": "success"
        }
        
        # Serialize the response to handle DateTime objects
        serialized_response = safe_json_response(response_data)
        
        return JSONResponse(
            content=serialized_response,
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={
                "error": str(e),
                "indicator": indicator_value,
                "status": "error"
            },
            status_code=500
        )
