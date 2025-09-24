"""
Enhanced RAG Service with LangGraph for intelligent routing between data stores.
Combines vector similarity search, graph traversal, and structured queries with advanced query types.
"""
import os
import logging
import re
from typing import Dict, List, Any
from typing_extensions import TypedDict
from collections import defaultdict

from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

# Import our existing clients and services
import sys
sys.path.append('/Users/akarna/Blackbox POC/src')

from shared.clients.neo4j_client import Neo4jKnowledgeGraphClient
from shared.clients.qdrant_client import QdrantVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class RAGState(TypedDict):
    """Enhanced state for the RAG agent."""
    query: str
    query_type: str
    query_intent: str
    vector_results: List[Dict]
    graph_results: List[Dict]
    indicator_results: List[Dict]
    pattern_results: List[Dict]
    campaign_results: List[Dict]
    timeline_results: List[Dict]
    final_response: str
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    search_strategy: str
    confidence_score: float


# Initialize clients with environment-aware configuration

# Get connection details from environment or use defaults
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "testpass")

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
qdrant_api_key = os.getenv("QDRANT_API_KEY", "nYvmqn8eYkq8cHeLGk5Vj_px3AzXGRkIkEbxt6virSJ-8uih0JJUQw")

neo4j_client = Neo4jKnowledgeGraphClient(
    uri=neo4j_uri,
    username=neo4j_username,
    password=neo4j_password
)
qdrant_client = QdrantVectorStore(
    host=qdrant_host,
    port=qdrant_port,
    collection_name="document_chunks",
    api_key=qdrant_api_key
)


@tool
def vector_search(query: str, limit: int = 10, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Search for similar content using vector similarity in Qdrant."""
    try:
        results = qdrant_client.search_similar(query, limit=limit, score_threshold=score_threshold)
        logger.info(f"Vector search found {len(results)} results for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return []


@tool
def graph_traversal(query: str, indicator_value: str = None, relationship_types: List[str] = None) -> List[Dict[str, Any]]:
    """Search the knowledge graph for entities and relationships."""
    try:
        if not neo4j_client.driver:
            return []
        
        results = []
        
        # Debug logging
        logger.info(f"Graph traversal called with query: '{query}', indicator_value: '{indicator_value}'")
        
        # If specific indicator provided, find related entities
        if indicator_value and indicator_value.strip():
            with neo4j_client.driver.session() as session:
                # First check if the indicator exists
                check_query = """
                MATCH (i:Indicator {normalized_value: $indicator_value})
                RETURN i
                LIMIT 1
                """
                check_result = session.run(check_query, indicator_value=indicator_value.lower())
                
                # Only proceed if indicator exists
                if check_result.single():
                    # Find the indicator and its relationships
                    cypher_query = """
                    MATCH (i:Indicator {normalized_value: $indicator_value})
                    MATCH (i)-[r]-(related)
                    RETURN i, r, related, type(r) as relationship_type
                    LIMIT 50
                    """
                    result = session.run(cypher_query, indicator_value=indicator_value.lower())
                    
                    for record in result:
                        results.append({
                            "entity": dict(record["i"]),
                            "relationship": dict(record["r"]) if record["r"] else None,
                            "related_entity": dict(record["related"]) if record["related"] else None,
                            "relationship_type": record["relationship_type"]
                        })
        
        # General graph search for entities matching the query
        else:
            with neo4j_client.driver.session() as session:
                # Check if this is a network analysis query with hops
                if "hops" in query.lower() and "domain" in query.lower():
                    # Extract domain and hop count from query
                    import re
                    hop_match = re.search(r'(\d+)\s*hops?', query.lower())
                    domain_match = re.search(r'domain\s+(\w+)', query.lower())
                    
                    if hop_match and domain_match:
                        hop_count = int(hop_match.group(1))
                        domain_name = domain_match.group(1)
                        
                        # Network analysis query with hop-based traversal
                        cypher_query = f"""
                        MATCH (start:Indicator {{indicator_type: 'domains'}})
                        WHERE toLower(start.value) CONTAINS toLower($domain_name)
                           OR toLower(start.normalized_value) CONTAINS toLower($domain_name)
                        MATCH path = (start)-[*1..{hop_count}]-(end:Indicator)
                        WHERE end <> start
                        RETURN DISTINCT end, length(path) as hop_distance, 
                               [node in nodes(path) | node.value] as path_nodes
                        ORDER BY hop_distance
                        LIMIT 100
                        """
                        result = session.run(cypher_query, domain_name=domain_name)
                        
                        for record in result:
                            results.append({
                                "entity": dict(record["end"]),
                                "hop_distance": record["hop_distance"],
                                "path_nodes": record["path_nodes"],
                                "relationship_type": "network_analysis"
                            })
                    else:
                        # Fallback to general search
                        cypher_query = """
                        MATCH (n)
                        WHERE toLower(n.value) CONTAINS toLower($search_query) 
                           OR toLower(n.normalized_value) CONTAINS toLower($search_query)
                           OR toLower(n.name) CONTAINS toLower($search_query)
                        OPTIONAL MATCH (n)-[r]-(related)
                        RETURN n, r, related, type(r) as relationship_type
                        LIMIT 50
                        """
                        result = session.run(cypher_query, search_query=query)
                        
                        for record in result:
                            results.append({
                                "entity": dict(record["n"]),
                                "relationship": dict(record["r"]) if record["r"] else None,
                                "related_entity": dict(record["related"]) if record["related"] else None,
                                "relationship_type": record["relationship_type"]
                            })
                else:
                    # Standard graph search
                    cypher_query = """
                    MATCH (n)
                    WHERE toLower(n.value) CONTAINS toLower($search_query) 
                       OR toLower(n.normalized_value) CONTAINS toLower($search_query)
                       OR toLower(n.name) CONTAINS toLower($search_query)
                    OPTIONAL MATCH (n)-[r]-(related)
                    RETURN n, r, related, type(r) as relationship_type
                    LIMIT 50
                    """
                    result = session.run(cypher_query, search_query=query)
                    
                    for record in result:
                        results.append({
                            "entity": dict(record["n"]),
                            "relationship": dict(record["r"]) if record["r"] else None,
                            "related_entity": dict(record["related"]) if record["related"] else None,
                            "relationship_type": record["relationship_type"]
                        })
        
        logger.info(f"Graph traversal found {len(results)} results for query: {query}")
        return results
        
    except Exception as e:
        logger.error(f"Graph traversal error: {e}")
        return []


@tool
def indicator_lookup(indicator_type: str, value: str = None) -> List[Dict[str, Any]]:
    """Look up specific indicators by type and optionally by value."""
    try:
        if not neo4j_client.driver:
            return []
        
        results = []
        with neo4j_client.driver.session() as session:
            if value:
                # Specific indicator lookup
                cypher_query = """
                MATCH (i:Indicator {indicator_type: $type, normalized_value: $value})
                OPTIONAL MATCH (i)-[:MENTIONED_IN]->(d:Document)
                RETURN i, d
                """
                result = session.run(cypher_query, type=indicator_type, value=value.lower())
            else:
                # All indicators of this type
                cypher_query = """
                MATCH (i:Indicator {indicator_type: $type})
                OPTIONAL MATCH (i)-[:MENTIONED_IN]->(d:Document)
                RETURN i, d
                LIMIT 100
                """
                result = session.run(cypher_query, type=indicator_type)
            
            for record in result:
                results.append({
                    "indicator": dict(record["i"]),
                    "document": dict(record["d"]) if record["d"] else None
                })
        
        logger.info(f"Indicator lookup found {len(results)} results for type: {indicator_type}")
        return results
        
    except Exception as e:
        logger.error(f"Indicator lookup error: {e}")
        return []




@tool
def pattern_detection(query: str, pattern_type: str = "clusters", filter_type: str = None) -> List[Dict[str, Any]]:
    """Detect patterns in the data such as clusters of related entities."""
    try:
        if not neo4j_client.driver:
            return []
        
        results = []
        with neo4j_client.driver.session() as session:
            if pattern_type == "clusters":
                # Find clusters of related indicators that appear in the same documents
                if filter_type:
                    # Filter for specific indicator type (e.g., social_media)
                    cypher_query = f"""
                    MATCH (i1:Indicator)-[:MENTIONED_IN]->(d:Document)<-[:MENTIONED_IN]-(i2:Indicator)
                    WHERE i1.id < i2.id AND i1.indicator_type = i2.indicator_type AND i1.indicator_type = '{filter_type}'
                    WITH i1.indicator_type as type, collect(DISTINCT i1) + collect(DISTINCT i2) as cluster_nodes, count(DISTINCT d) as doc_count
                    WHERE size(cluster_nodes) >= 3
                    RETURN type, cluster_nodes, size(cluster_nodes) as cluster_size, doc_count
                    ORDER BY cluster_size DESC
                    LIMIT 20
                    """
                else:
                    # General cluster detection
                    cypher_query = """
                    MATCH (i1:Indicator)-[:MENTIONED_IN]->(d:Document)<-[:MENTIONED_IN]-(i2:Indicator)
                    WHERE i1.id < i2.id AND i1.indicator_type = i2.indicator_type
                    WITH i1.indicator_type as type, collect(DISTINCT i1) + collect(DISTINCT i2) as cluster_nodes, count(DISTINCT d) as doc_count
                    WHERE size(cluster_nodes) >= 3
                    RETURN type, cluster_nodes, size(cluster_nodes) as cluster_size, doc_count
                    ORDER BY cluster_size DESC
                    LIMIT 20
                    """
                result = session.run(cypher_query)
                
                for record in result:
                    # Convert Neo4j nodes to dictionaries and remove duplicates by id
                    cluster_nodes = record["cluster_nodes"]
                    seen_ids = set()
                    unique_nodes = []
                    for node in cluster_nodes:
                        node_dict = dict(node)
                        if node_dict.get("id") not in seen_ids:
                            seen_ids.add(node_dict.get("id"))
                            unique_nodes.append(node_dict)
                    
                    results.append({
                        "pattern_type": "cluster",
                        "indicator_type": record["type"],
                        "cluster_size": record["cluster_size"],
                        "doc_count": record.get("doc_count", 0),
                        "nodes": unique_nodes[:10],  # Limit for response size
                        "description": f"Cluster of {record['cluster_size']} {record['type']} indicators across {record.get('doc_count', 0)} documents"
                    })
            
            elif pattern_type == "temporal":
                # Find temporal patterns in indicator creation
                cypher_query = """
                MATCH (i:Indicator)
                WHERE i.created_at IS NOT NULL
                WITH date(i.created_at) as creation_date, collect(i) as indicators
                WHERE size(indicators) >= 2
                RETURN creation_date, indicators, size(indicators) as count
                ORDER BY creation_date DESC
                LIMIT 15
                """
                result = session.run(cypher_query)
                
                for record in result:
                    results.append({
                        "pattern_type": "temporal",
                        "date": str(record["creation_date"]),
                        "indicator_count": record["count"],
                        "indicators": [dict(ind) for ind in record["indicators"][:5]],
                        "description": f"Temporal cluster: {record['count']} indicators on {record['creation_date']}"
                    })
        
        logger.info(f"Pattern detection found {len(results)} patterns")
        return results
        
    except Exception as e:
        logger.error(f"Pattern detection error: {e}")
        return []


@tool
def campaign_analysis(query: str) -> List[Dict[str, Any]]:
    """Analyze which indicators appear across multiple campaigns or documents."""
    try:
        if not neo4j_client.driver:
            return []
        
        results = []
        with neo4j_client.driver.session() as session:
            # Find indicators that appear in multiple documents
            cypher_query = """
            MATCH (i:Indicator)-[:MENTIONED_IN]->(d:Document)
            WITH i, collect(d) as documents, count(d) as doc_count
            WHERE doc_count >= 2
            RETURN i, documents, doc_count
            ORDER BY doc_count DESC
            LIMIT 50
            """
            result = session.run(cypher_query)
            
            for record in result:
                indicator = dict(record["i"])
                documents = [dict(doc) for doc in record["documents"]]
                
                results.append({
                    "indicator": indicator,
                    "cross_document_count": record["doc_count"],
                    "source_documents": documents,
                    "description": f"Indicator appears in {record['doc_count']} documents",
                    "significance": "high" if record["doc_count"] >= 3 else "medium"
                })
        
        logger.info(f"Campaign analysis found {len(results)} cross-campaign indicators")
        return results
        
    except Exception as e:
        logger.error(f"Campaign analysis error: {e}")
        return []


@tool
def timeline_query(query: str, time_range: str = "all") -> List[Dict[str, Any]]:
    """Show indicator relationships over time with enhanced temporal analysis."""
    try:
        if not neo4j_client.driver:
            return []
        
        results = []
        with neo4j_client.driver.session() as session:
            # Enhanced query to get indicators with relationships and temporal data
            cypher_query = """
            MATCH (i:Indicator)
            WHERE i.created_at IS NOT NULL
            OPTIONAL MATCH (i)-[r]-(related)
            RETURN i, r, related, i.created_at as timestamp,
                   i.indicator_type as type, i.value as value
            ORDER BY timestamp DESC
            LIMIT 200
            """
            result = session.run(cypher_query)
            
            timeline_data = defaultdict(list)
            relationship_network = defaultdict(set)
            
            for record in result:
                indicator = dict(record["i"])
                relationship = dict(record["r"]) if record["r"] else None
                related_entity = dict(record["related"]) if record["related"] else None
                timestamp = record["timestamp"]
                
                # Extract date from timestamp
                if timestamp:
                    if hasattr(timestamp, 'date'):
                        date_key = str(timestamp.date())
                    else:
                        # Handle string timestamps
                        date_key = str(timestamp)[:10] if len(str(timestamp)) >= 10 else "unknown"
                else:
                    date_key = "unknown"
                
                timeline_entry = {
                    "indicator": {
                        "value": indicator.get("value", ""),
                        "type": indicator.get("indicator_type", ""),
                        "confidence": indicator.get("confidence_score", 0.0)
                    },
                    "relationship": {
                        "type": relationship.get("type", "") if relationship else None,
                        "properties": dict(relationship) if relationship else None
                    },
                    "related_entity": {
                        "value": related_entity.get("value", "") if related_entity else None,
                        "type": related_entity.get("indicator_type", "") if related_entity else None
                    },
                    "timestamp": str(timestamp) if timestamp else None,
                    "source_document": indicator.get("source_document", "")
                }
                
                timeline_data[date_key].append(timeline_entry)
                
                # Build relationship network for analysis
                if related_entity:
                    indicator_key = f"{indicator.get('value', '')}_{indicator.get('indicator_type', '')}"
                    related_key = f"{related_entity.get('value', '')}_{related_entity.get('indicator_type', '')}"
                    relationship_network[indicator_key].add(related_key)
            
            # Convert to enhanced timeline format with relationship analysis
            for date, entries in sorted(timeline_data.items(), key=lambda x: x[0], reverse=True):
                # Analyze relationships for this date
                relationship_count = sum(1 for entry in entries if entry["relationship"]["type"])
                indicator_types = {}
                for entry in entries:
                    ind_type = entry["indicator"]["type"]
                    indicator_types[ind_type] = indicator_types.get(ind_type, 0) + 1
                
                # Find most connected indicators
                connections = defaultdict(int)
                for entry in entries:
                    if entry["related_entity"]["value"]:
                        key = f"{entry['indicator']['value']}_{entry['indicator']['type']}"
                        connections[key] += 1
                
                most_connected = sorted(connections.items(), key=lambda x: x[1], reverse=True)[:5]
                
                results.append({
                    "date": date,
                    "events": entries,
                    "event_count": len(entries),
                    "relationship_count": relationship_count,
                    "indicator_types": indicator_types,
                    "most_connected": [{"indicator": k.split("_")[0], "type": k.split("_")[1], "connections": v} 
                                     for k, v in most_connected],
                    "description": f"{len(entries)} indicators with {relationship_count} relationships on {date}",
                    "analysis": {
                        "total_indicators": len(entries),
                        "total_relationships": relationship_count,
                        "indicator_type_distribution": indicator_types,
                        "most_active_indicators": most_connected[:3]
                    }
                })
        
        logger.info(f"Enhanced timeline query found {len(results)} timeline entries")
        return results
        
    except Exception as e:
        logger.error(f"Timeline query error: {e}")
        return []


@tool
def semantic_search(query: str, search_type: str = "hybrid") -> Dict[str, Any]:
    """Enhanced semantic search combining multiple search strategies."""
    try:
        results = {
            "vector_results": [],
            "graph_results": [],
            "indicator_results": [],
            "search_strategy": search_type
        }
        
        # Vector search
        vector_results = vector_search.invoke(query, limit=10)
        results["vector_results"] = vector_results
        
        # Graph search
        graph_results = graph_traversal.invoke(query)
        results["graph_results"] = graph_results
        
        # Simplified - no indicator extraction in semantic search
        
        # Calculate confidence score based on result quality
        total_results = len(vector_results) + len(graph_results)
        confidence = min(1.0, total_results / 20.0)  # Normalize to 0-1
        
        results["confidence_score"] = confidence
        results["total_results"] = total_results
        
        logger.info(f"Semantic search found {total_results} total results with confidence {confidence:.2f}")
        return results
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return {"error": str(e), "confidence_score": 0.0}


class RAGAgent:
    """Enhanced LangGraph-based RAG agent for intelligent routing between data stores."""
    
    def __init__(self):
        # RAG agent without LLM - uses direct data retrieval only
        
        # Tool set with only used tools
        self.tools = [
            vector_search, 
            graph_traversal, 
            indicator_lookup
        ]
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        def classify_query(state: RAGState) -> RAGState:
            """Enhanced query classification with better intent prioritization."""
            query = state["query"].lower()

            query_intent = "general"
            query_type = "hybrid_search"
            search_strategy = "balanced"
            confidence_score = 0.5

            # Intent detection with priority scoring
            intent_patterns = {
                "timeline": ["timeline", "over time", "chronological", "when", "date", "history", "relationships over time", "show indicator relationships over time"],
                "pattern_detection": ["cluster", "pattern", "group", "similar", "related accounts"],
                "campaign_analysis": ["across", "multiple", "documents", "appears in"],
                "network_analysis": ["network", "connected", "relationship", "graph", "hops"],
                "indicator_lookup": ["find", "search for", "get", "show me", "list"],
                "semantic": ["what", "how", "why", "explain", "describe", "tell me about"],
                "context": ["context", "full context", "details", "background", "information about"],
            }

            best_intent = "general"
            best_score = 0.0
            for intent, patterns in intent_patterns.items():
                for pattern in patterns:
                    if pattern in query:
                        score = 1.0 if intent == "timeline" else 0.8  # timeline wins
                        if score > best_score:
                            best_intent = intent
                            best_score = score

            query_intent = best_intent
            confidence_score = min(1.0, confidence_score + best_score)

            # Refined query type rules - prioritize pattern detection for clustering queries
            if any(word in query for word in ["cluster", "pattern", "group", "clusters of", "related accounts", "find clusters"]):
                query_type = "pattern_search"
                search_strategy = "pattern_focused"
            elif any(word in query for word in ["timeline", "time", "date", "chronological", "over time", "relationships over time", "show indicator relationships over time"]):
                query_type = "timeline_search"
                search_strategy = "temporal"
            elif any(word in query for word in ["relationship", "connected", "network", "graph", "hops"]):
                query_type = "graph_search"
                search_strategy = "graph_focused"
            elif any(word in query for word in ["indicator", "domain", "domains", "url", "urls", "email", "emails", "phone", "phones", "ip", "ips", "social", "which indicators", "indicators appear", "find all"]):
                query_type = "indicator_search"
                search_strategy = "targeted"
            elif any(word in query for word in ["campaign", "across", "multiple"]) and not any(word in query for word in ["domain", "domains", "url", "urls", "email", "emails", "phone", "phones", "ip", "ips", "social"]):
                # Distinguish between campaign information queries and cross-campaign analysis
                if any(word in query for word in ["what", "which", "how", "why", "when", "where", "who", "target", "targets", "against", "about"]):
                    # Campaign information query - use vector search for semantic content
                    query_type = "vector_search"
                    search_strategy = "vector_focused"
                else:
                    # Cross-campaign analysis query
                    query_type = "campaign_search"
                    search_strategy = "cross_document"
            elif any(word in query for word in ["similar", "semantic", "content", "text"]):
                query_type = "vector_search"
                search_strategy = "vector_focused"
            else:
                query_type = "hybrid_search"
                search_strategy = "balanced"

            state["query_type"] = query_type
            state["query_intent"] = query_intent
            state["search_strategy"] = search_strategy
            state["confidence_score"] = confidence_score

            logger.info(f"Query classified as: {query_type} (intent: {query_intent}, strategy: {search_strategy})")
            return state
        
        def route_to_pattern_search(state: RAGState) -> RAGState:
            """Route to pattern detection search with social media filtering."""
            query = state["query"].lower()
            
            # Check if query is specifically about social media
            if any(word in query for word in ["social media", "social", "twitter", "facebook", "instagram", "linkedin", "tiktok", "youtube"]):
                # Filter for social media patterns only
                results = pattern_detection.invoke({"query": state["query"], "pattern_type": "clusters", "filter_type": "social_media"})
            else:
                # General pattern detection
                results = pattern_detection.invoke({"query": state["query"], "pattern_type": "clusters"})
            
            state["pattern_results"] = results
            return state
        
        def route_to_campaign_search(state: RAGState) -> RAGState:
            """Route to campaign analysis search."""
            results = campaign_analysis.invoke({"query": state["query"]})
            state["campaign_results"] = results
            return state
        
        def route_to_timeline_search(state: RAGState) -> RAGState:
            """Route to timeline query search."""
            results = timeline_query.invoke({"query": state["query"]})
            state["timeline_results"] = results
            return state
        
        def hybrid_search(state: RAGState) -> RAGState:
            """Simplified hybrid search using semantic search for comprehensive results."""
            # Use semantic search for comprehensive results
            semantic_results = semantic_search.invoke({"query": state["query"], "search_type": "hybrid"})
            state["vector_results"] = semantic_results.get("vector_results", [])
            state["graph_results"] = semantic_results.get("graph_results", [])
            state["indicator_results"] = semantic_results.get("indicator_results", [])
            state["confidence_score"] = semantic_results.get("confidence_score", 0.5)
            
            return state
        
        def generate_response(state: RAGState) -> RAGState:
            """Enhanced response generation with support for all query types."""
            # Prepare context from all search results
            context_parts = []
            
            # Vector search results
            if state.get("vector_results"):
                context_parts.append("Vector Search Results:")
                for i, result in enumerate(state["vector_results"][:5]):
                    content = result.get('payload', {}).get('content', '')
                    context_parts.append(f"{i+1}. {content[:200]}...")
            
            # Graph search results
            if state.get("graph_results"):
                context_parts.append("\nGraph Search Results:")
                for i, result in enumerate(state["graph_results"][:5]):
                    if 'indicator' in result:
                        context_parts.append(f"{i+1}. Indicator: {result['indicator'].get('value', '')}")
                    elif 'entity' in result:
                        context_parts.append(f"{i+1}. Entity: {result['entity'].get('value', '')}")
            
            # Indicator search results
            if state.get("indicator_results"):
                context_parts.append("\nIndicator Search Results:")
                for i, result in enumerate(state["indicator_results"][:5]):
                    indicator = result.get('indicator', {})
                    context_parts.append(f"{i+1}. {indicator.get('indicator_type', '')}: {indicator.get('value', '')}")
            
            # Pattern detection results
            if state.get("pattern_results"):
                context_parts.append("\nPattern Detection Results:")
                for i, result in enumerate(state["pattern_results"][:3]):
                    context_parts.append(f"{i+1}. {result.get('description', '')}")
            
            # Campaign analysis results
            if state.get("campaign_results"):
                context_parts.append("\nCampaign Analysis Results:")
                for i, result in enumerate(state["campaign_results"][:3]):
                    context_parts.append(f"{i+1}. {result.get('description', '')}")
            
            # Timeline results
            if state.get("timeline_results"):
                context_parts.append("\nTimeline Analysis Results:")
                for i, result in enumerate(state["timeline_results"][:3]):
                    context_parts.append(f"{i+1}. {result.get('description', '')}")
            
            context = "\n".join(context_parts)
            
            # Generate response using fallback method
            state["final_response"] = self._generate_fallback_response(state, context)
            
            return state
        
        
        
        def route_to_indicator_search(state: RAGState) -> RAGState:
            """Route to indicator search with enhanced cross-campaign analysis."""
            query = state["query"].lower()
            
            # Check if this is a cross-campaign analysis query
            if any(word in query for word in ["across", "multiple", "campaigns", "which indicators appear"]):
                # Enhanced cross-campaign indicator analysis
                try:
                    if not neo4j_client.driver:
                        state["indicator_results"] = []
                        return state
                    
                    cross_campaign_indicators = []
                    with neo4j_client.driver.session() as session:
                        # Find indicators that appear in multiple documents with detailed analysis
                        cypher_query = """
                        MATCH (i:Indicator)-[:MENTIONED_IN]->(d:Document)
                        WITH i, collect(DISTINCT d) as documents, count(DISTINCT d) as doc_count
                        WHERE doc_count >= 2
                        RETURN i, documents, doc_count
                        ORDER BY doc_count DESC, i.value ASC
                        LIMIT 50
                        """
                        result = session.run(cypher_query)
                        
                        for record in result:
                            indicator = dict(record["i"])
                            documents = [dict(doc) for doc in record["documents"]]
                            doc_count = record["doc_count"]
                            
                            # Create enhanced indicator result with cross-campaign analysis
                            cross_campaign_indicators.append({
                                "indicator": {
                                    "value": indicator.get("value", ""),
                                    "type": indicator.get("indicator_type", ""),
                                    "confidence": indicator.get("confidence_score", 0.0),
                                    "created_at": indicator.get("created_at", ""),
                                    "source_document": indicator.get("source_document", "")
                                },
                                "cross_campaign_analysis": {
                                    "appears_in_documents": doc_count,
                                    "document_names": [doc.get("filename", "") for doc in documents],
                                    "significance": "high" if doc_count >= 3 else "medium",
                                    "campaign_coverage": f"Appears in {doc_count} out of {len(documents)} documents"
                                },
                                "source_documents": documents,
                                "description": f"Indicator '{indicator.get('value', '')}' appears across {doc_count} campaigns/documents"
                            })
                    
                    state["indicator_results"] = cross_campaign_indicators
                    logger.info(f"Cross-campaign indicator analysis found {len(cross_campaign_indicators)} indicators")
                    
                except Exception as e:
                    logger.error(f"Cross-campaign indicator analysis error: {e}")
                    state["indicator_results"] = []
            else:
                # Regular indicator search
                indicator_patterns = {
                    "domains": r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b',
                    "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    "ip_addresses": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                    "urls": r'https?://[^\s<>"{}|\\^`\[\]]+'
                }
                
                found_indicators = []
                for indicator_type, pattern in indicator_patterns.items():
                    matches = re.findall(pattern, state["query"], re.IGNORECASE)
                    for match in matches:
                        found_indicators.append((indicator_type, match))
                
                # Check if query is asking for specific indicator types (with or without entity association)
                if not found_indicators and any(word in query for word in ["domains", "emails", "urls", "ip", "social media", "phone"]):
                    # Extract the entity name and indicator type from the query
                    entity_name = None
                    indicator_type = None
                    
                    # Extract entity name (everything after "associated with" or "related to")
                    if "associated with" in query:
                        entity_name = query.split("associated with")[-1].strip()
                    elif "related to" in query:
                        entity_name = query.split("related to")[-1].strip()
                    elif "for" in query:
                        entity_name = query.split("for")[-1].strip()
                    
                    # Normalize special characters for better matching
                    if entity_name:
                        # Replace common special characters with their ASCII equivalents
                        entity_name = entity_name.replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
                        entity_name = entity_name.replace("é", "e").replace("è", "e").replace("ê", "e").replace("ë", "e")
                        entity_name = entity_name.replace("à", "a").replace("á", "a").replace("â", "a").replace("ã", "a")
                        entity_name = entity_name.replace("ç", "c").replace("ñ", "n")
                        entity_name = entity_name.strip()
                    
                    # Determine indicator type
                    if "domains" in query:
                        indicator_type = "domains"
                    elif "emails" in query:
                        indicator_type = "emails"
                    elif "urls" in query:
                        indicator_type = "urls"
                    elif "ip" in query:
                        indicator_type = "ip_addresses"
                    elif "social media" in query:
                        indicator_type = "social_media"
                    elif "phone" in query:
                        indicator_type = "phone_numbers"
                    
                    if indicator_type:
                        # Search for indicators of the specified type
                        try:
                            if not neo4j_client.driver:
                                state["indicator_results"] = []
                                return state
                            
                            results = []
                            with neo4j_client.driver.session() as session:
                                if entity_name:
                                    # Find indicators of the specified type that are associated with the entity
                                    cypher_query = """
                                    MATCH (i:Indicator {indicator_type: $indicator_type})
                                    WHERE toLower(i.value) CONTAINS toLower($entity_name) 
                                       OR toLower(i.source_document) CONTAINS toLower($entity_name)
                                       OR EXISTS {
                                           MATCH (i)-[r]-(related)
                                           WHERE toLower(related.value) CONTAINS toLower($entity_name)
                                              OR toLower(related.name) CONTAINS toLower($entity_name)
                                       }
                                       OR EXISTS {
                                           MATCH (i)-[:MENTIONED_IN]->(d:Document)
                                           WHERE toLower(d.filename) CONTAINS toLower($entity_name)
                                       }
                                       OR EXISTS {
                                           MATCH (i)-[:MENTIONED_IN]->(d:Document)<-[:MENTIONED_IN]-(related:Indicator)
                                           WHERE toLower(related.value) CONTAINS toLower($entity_name)
                                       }
                                    OPTIONAL MATCH (i)-[:MENTIONED_IN]->(d:Document)
                                    RETURN i, d
                                    LIMIT 50
                                    """
                                    result = session.run(cypher_query, indicator_type=indicator_type, entity_name=entity_name)
                                    description = f"{indicator_type} associated with {entity_name}"
                                else:
                                    # Find all indicators of the specified type
                                    cypher_query = """
                                    MATCH (i:Indicator {indicator_type: $indicator_type})
                                    OPTIONAL MATCH (i)-[:MENTIONED_IN]->(d:Document)
                                    RETURN i, d
                                    LIMIT 50
                                    """
                                    result = session.run(cypher_query, indicator_type=indicator_type)
                                    description = f"All {indicator_type}"
                                
                                for record in result:
                                    results.append({
                                        "indicator": dict(record["i"]),
                                        "document": dict(record["d"]) if record["d"] else None,
                                        "description": description
                                    })
                            
                            state["indicator_results"] = results
                            if entity_name:
                                logger.info(f"Entity-associated indicator search found {len(results)} {indicator_type} for {entity_name}")
                            else:
                                logger.info(f"General indicator search found {len(results)} {indicator_type}")
                            
                        except Exception as e:
                            logger.error(f"Indicator search error: {e}")
                            state["indicator_results"] = []
                    else:
                        state["indicator_results"] = []
                else:
                    # Search for found indicators
                    results = []
                    for indicator_type, value in found_indicators:
                        indicator_results = indicator_lookup.invoke(indicator_type, value)
                        results.extend(indicator_results)
                    
                    state["indicator_results"] = results
            
            return state
        
        def route_after_classification(state: RAGState) -> str:
            """Enhanced routing based on query type and intent."""
            query_type = state["query_type"]
            
            # Route based on query type
            if query_type == "vector_search":
                return "vector_search"
            elif query_type == "graph_search":
                return "graph_search"
            elif query_type == "indicator_search":
                return "indicator_search"
            elif query_type == "pattern_search":
                return "pattern_search"
            elif query_type == "campaign_search":
                return "campaign_search"
            elif query_type == "timeline_search":
                return "timeline_search"
            else:
                return "hybrid_search"
        
        # Build the enhanced graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("classify", classify_query)
        def vector_search_route(state: RAGState) -> RAGState:
            results = vector_search.invoke(state["query"], limit=15)
            state["vector_results"] = results
            return state
        
        def graph_search_route(state: RAGState) -> RAGState:
            results = graph_traversal.invoke(state["query"])
            state["graph_results"] = results
            return state
        
        workflow.add_node("vector_search", vector_search_route)
        workflow.add_node("graph_search", graph_search_route)
        workflow.add_node("indicator_search", route_to_indicator_search)
        workflow.add_node("pattern_search", route_to_pattern_search)
        workflow.add_node("campaign_search", route_to_campaign_search)
        workflow.add_node("timeline_search", route_to_timeline_search)
        workflow.add_node("hybrid_search", hybrid_search)
        workflow.add_node("generate_response", generate_response)
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "classify",
            route_after_classification,
            {
                "vector_search": "vector_search",
                "graph_search": "graph_search", 
                "indicator_search": "indicator_search",
                "pattern_search": "pattern_search",
                "campaign_search": "campaign_search",
                "timeline_search": "timeline_search",
                "hybrid_search": "hybrid_search"
            }
        )
        
        # All search nodes go to generate_response
        workflow.add_edge("vector_search", "generate_response")
        workflow.add_edge("graph_search", "generate_response")
        workflow.add_edge("indicator_search", "generate_response")
        workflow.add_edge("pattern_search", "generate_response")
        workflow.add_edge("campaign_search", "generate_response")
        workflow.add_edge("timeline_search", "generate_response")
        workflow.add_edge("hybrid_search", "generate_response")
        
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        return workflow.compile()
    
    def _generate_fallback_response(self, state: RAGState, context: str) -> str:
        """Generate response based on search results without LLM."""
        response_parts = [f"Query: {state['query']}"]
        response_parts.append(f"Query Type: {state.get('query_type', 'unknown')}")
        response_parts.append(f"Search Strategy: {state.get('search_strategy', 'unknown')}")
        
        # Vector results
        if state.get("vector_results"):
            response_parts.append(f"\nFound {len(state['vector_results'])} relevant document chunks:")
            for i, result in enumerate(state["vector_results"][:3]):
                content = result.get('payload', {}).get('content', '')[:100]
                response_parts.append(f"  {i+1}. {content}...")
        
        # Graph results
        if state.get("graph_results"):
            response_parts.append(f"\nFound {len(state['graph_results'])} graph relationships:")
            for i, result in enumerate(state["graph_results"][:3]):
                if 'indicator' in result:
                    response_parts.append(f"  {i+1}. Indicator: {result['indicator'].get('value', '')}")
                elif 'entity' in result:
                    response_parts.append(f"  {i+1}. Entity: {result['entity'].get('value', '')}")
        
        # Indicator results
        if state.get("indicator_results"):
            response_parts.append(f"\nFound {len(state['indicator_results'])} indicators:")
            for i, result in enumerate(state["indicator_results"][:3]):
                indicator = result.get('indicator', {})
                response_parts.append(f"  {i+1}. {indicator.get('indicator_type', '')}: {indicator.get('value', '')}")
        
        # Pattern results
        if state.get("pattern_results"):
            response_parts.append(f"\nFound {len(state['pattern_results'])} patterns:")
            for i, result in enumerate(state["pattern_results"][:3]):
                response_parts.append(f"  {i+1}. {result.get('description', '')}")
        
        # Campaign results
        if state.get("campaign_results"):
            response_parts.append(f"\nFound {len(state['campaign_results'])} cross-campaign indicators:")
            for i, result in enumerate(state["campaign_results"][:3]):
                response_parts.append(f"  {i+1}. {result.get('description', '')}")
        
        # Timeline results
        if state.get("timeline_results"):
            response_parts.append(f"\nFound {len(state['timeline_results'])} timeline events:")
            for i, result in enumerate(state["timeline_results"][:3]):
                response_parts.append(f"  {i+1}. {result.get('description', '')}")
        
        # Check if any results found
        has_results = any([
            state.get("vector_results"),
            state.get("graph_results"),
            state.get("indicator_results"),
            state.get("pattern_results"),
            state.get("campaign_results"),
            state.get("timeline_results")
        ])
        
        if not has_results:
            response_parts.append("\nNo relevant results found for this query.")
        
        return "\n".join(response_parts)
    
    def search(self, query: str) -> Dict[str, Any]:
        """Enhanced intelligent search using the LangGraph agent."""
        try:
            # Initialize enhanced state
            initial_state = RAGState(
                query=query,
                query_type="",
                query_intent="",
                vector_results=[],
                graph_results=[],
                indicator_results=[],
                pattern_results=[],
                campaign_results=[],
                timeline_results=[],
                final_response="",
                context={},
                metadata={},
                search_strategy="",
                confidence_score=0.0
            )
            
            # Run the enhanced graph
            result = self.graph.invoke(initial_state)
            
            return {
                "query": query,
                "response": result.get("final_response", ""),
                "vector_results": result.get("vector_results", []),
                "graph_results": result.get("graph_results", []),
                "indicator_results": result.get("indicator_results", []),
                "pattern_results": result.get("pattern_results", []),
                "campaign_results": result.get("campaign_results", []),
                "timeline_results": result.get("timeline_results", []),
                "query_type": result.get("query_type", ""),
                "query_intent": result.get("query_intent", ""),
                "search_strategy": result.get("search_strategy", ""),
                "confidence_score": result.get("confidence_score", 0.0),
                "metadata": {
                    "total_vector_results": len(result.get("vector_results", [])),
                    "total_graph_results": len(result.get("graph_results", [])),
                    "total_indicator_results": len(result.get("indicator_results", [])),
                    "total_pattern_results": len(result.get("pattern_results", [])),
                    "total_campaign_results": len(result.get("campaign_results", [])),
                    "total_timeline_results": len(result.get("timeline_results", []))
                }
            }
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "error": str(e)
            }
    
    def get_indicators_by_type(self, indicator_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all indicators of a specific type."""
        return indicator_lookup.invoke(indicator_type)
    
    def get_indicator_context(self, indicator_value: str) -> Dict[str, Any]:
        """Get full context for a specific indicator."""
        # Get graph relationships
        graph_results = graph_traversal.invoke({"query": "", "indicator_value": indicator_value})
        
        # Get vector search results for context
        vector_results = vector_search.invoke({"query": indicator_value})
        
        return {
            "indicator": indicator_value,
            "graph_context": graph_results,
            "vector_context": vector_results,
            "total_relationships": len(graph_results),
            "total_context_chunks": len(vector_results)
        }
    
    def get_relationships(self, indicator_value: str) -> List[Dict[str, Any]]:
        """Get relationships for a specific indicator."""
        return graph_traversal.invoke({"query": "", "indicator_value": indicator_value})
    


# Global RAG agent instance
rag_agent = RAGAgent()
