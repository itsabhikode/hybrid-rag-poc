import os
import sys
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from shared.services.rag_service import rag_agent
from shared.utils.json_serializer import safe_json_response, format_indicator_response, format_document_response

router = APIRouter()


@router.get("/indicators/{indicator_type}")
async def get_indicator_type(
    indicator_type: str,
    limit: int = Query(100, description="Maximum number of indicators to return"),
    value: Optional[str] = Query(None, description="Specific indicator value to search for")
):
    """
    Retrieve specific indicator types from the knowledge graph.
    Supports filtering by specific values.
    """
    try:
        # Validate indicator type - use correct plural forms from database
        valid_types = ["domains", "urls", "emails", "phone_numbers", "ip_addresses", "social_media", "tracking_ids"]
        if indicator_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid indicator type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Get indicators using RAG agent
        if value:
            # Search for specific indicator value
            results = rag_agent.get_indicators_by_type(indicator_type)
            # Filter by value
            filtered_results = [
                r for r in results 
                if value.lower() in r.get("indicator", {}).get("value", "").lower()
            ]
        else:
            # Get all indicators of this type
            results = rag_agent.get_indicators_by_type(indicator_type, limit=limit)
            filtered_results = results[:limit]
        
        # Format response with proper DateTime serialization
        indicators = []
        for result in filtered_results:
            indicator_data = result.get("indicator", {})
            document_data = result.get("document", {})
            
            indicator = format_indicator_response(indicator_data)
            
            if document_data:
                indicator["source_document"] = format_document_response(document_data)
            else:
                indicator["source_document"] = None
            
            indicators.append(indicator)
        
        response_data = {
            "indicator_type": indicator_type,
            "total_count": len(indicators),
            "indicators": indicators,
            "filters": {
                "value": value,
                "limit": limit
            },
            "status": "success"
        }
        
        # Serialize the response to handle any remaining DateTime objects
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
                "indicator_type": indicator_type,
                "status": "error"
            },
            status_code=500
        )
