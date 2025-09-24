import os
import sys

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from shared.services.rag_service import rag_agent
from shared.utils.json_serializer import safe_json_response

router = APIRouter()


@router.get("/context/{indicator_value}")
async def get_full_context_of_indicator(indicator_value: str):
    """
    Get full context for a specific indicator including:
    - Graph relationships and connections
    - Vector similarity context
    - Source documents and metadata
    """
    try:
        if not indicator_value.strip():
            raise HTTPException(status_code=400, detail="Indicator value is required")
        
        # Get comprehensive context using RAG agent
        context = rag_agent.get_indicator_context(indicator_value)
        
        # Format the response
        formatted_context = {
            "indicator": indicator_value,
            "graph_context": {
                "relationships": context.get("graph_context", []),
                "total_relationships": context.get("total_relationships", 0)
            },
            "vector_context": {
                "similar_chunks": context.get("vector_context", []),
                "total_context_chunks": context.get("total_context_chunks", 0)
            },
            "summary": {
                "total_connections": context.get("total_relationships", 0),
                "total_context_documents": context.get("total_context_chunks", 0),
                "has_graph_connections": context.get("total_relationships", 0) > 0,
                "has_vector_context": context.get("total_context_chunks", 0) > 0
            }
        }
        
        # Add detailed relationship analysis
        relationships = context.get("graph_context", [])
        if relationships:
            relationship_types = {}
            connected_entities = {}
            
            for rel in relationships:
                rel_type = rel.get("relationship_type", "unknown")
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                
                related_entity = rel.get("related_entity", {})
                if related_entity:
                    entity_type = related_entity.get("labels", ["unknown"])[0] if related_entity.get("labels") else "unknown"
                    connected_entities[entity_type] = connected_entities.get(entity_type, 0) + 1
            
            formatted_context["analysis"] = {
                "relationship_types": relationship_types,
                "connected_entity_types": connected_entities,
                "most_common_relationship": max(relationship_types.items(), key=lambda x: x[1])[0] if relationship_types else None,
                "most_connected_entity_type": max(connected_entities.items(), key=lambda x: x[1])[0] if connected_entities else None
            }
        
        response_data = {
            "indicator": indicator_value,
            "context": formatted_context,
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
