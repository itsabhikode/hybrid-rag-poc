import os
import sys

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from shared.utils.json_serializer import safe_json_response

router = APIRouter()



@router.get("/search")
async def search_get(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results", ge=1, le=100),
    score_threshold: float = Query(0.7, description="Score threshold for results", ge=0.0, le=1.0)
):
    """
    GET endpoint for intelligent search using LangGraph agent for routing between data stores.
    Supports semantic search, indicator lookup, graph traversal, pattern detection, 
    campaign analysis, and timeline queries.
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Use LangGraph agent for intelligent routing
        from shared.services.rag_service import rag_agent
        
        # Execute the search using the RAG agent
        search_result = rag_agent.search(query)
        
        # Format the response with intelligent routing results
        response_data = {
            "query": query,
            "search_type": search_result.get("search_type", "intelligent_routing"),
            "query_classification": {
                "query_type": search_result.get("query_type", "unknown"),
                "query_intent": search_result.get("query_intent", "unknown"),
                "search_strategy": search_result.get("search_strategy", "unknown")
            },
            "response": search_result.get("final_response", f"Found results for '{query}'"),
            "confidence_score": search_result.get("confidence_score", 0.8),
            "results": {
                "vector_results": search_result.get("vector_results", []),
                "graph_results": search_result.get("graph_results", []),
                "indicator_results": search_result.get("indicator_results", []),
                "pattern_results": search_result.get("pattern_results", []),
                "campaign_results": search_result.get("campaign_results", []),
                "timeline_results": search_result.get("timeline_results", [])
            },
            "metadata": {
                "total_vector_results": len(search_result.get("vector_results", [])),
                "total_graph_results": len(search_result.get("graph_results", [])),
                "total_indicator_results": len(search_result.get("indicator_results", [])),
                "total_pattern_results": len(search_result.get("pattern_results", [])),
                "total_campaign_results": len(search_result.get("campaign_results", [])),
                "total_timeline_results": len(search_result.get("timeline_results", []))
            },
            "parameters": {
                "limit": limit,
                "score_threshold": score_threshold
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
                "query": query,
                "status": "error"
            },
            status_code=500
        )