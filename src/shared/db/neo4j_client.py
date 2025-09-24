"""
Neo4j Knowledge Graph Client

This module provides a client for interacting with the Neo4j knowledge graph,
including CRUD operations, complex queries, and data ingestion from the document processing pipeline.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import json
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError, TransientError

from .neo4j_models import (
    BaseNode, Document, Indicator, Campaign, ThreatActor, Chunk,
    Relationship, DocumentIndicatorRelationship, IndicatorIndicatorRelationship,
    CampaignIndicatorRelationship, CampaignThreatActorRelationship,
    IndicatorType, ThreatLevel, CampaignType, RelationshipType,
    KnowledgeGraphSchema, create_document_node, create_indicator_node,
    create_campaign_node, create_threat_actor_node, create_chunk_node
)

logger = logging.getLogger(__name__)


class Neo4jKnowledgeGraphClient:
    """Client for Neo4j knowledge graph operations."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", password: str = "testpass",
                 database: str = "neo4j"):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI
            username: Database username
            password: Database password
            database: Database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _get_session(self) -> Session:
        """Get a new database session."""
        if not self.driver:
            self._connect()
        return self.driver.session(database=self.database)
    
    def setup_schema(self) -> Dict[str, Any]:
        """Set up the knowledge graph schema with indexes and constraints."""
        results = {"indexes": [], "constraints": [], "errors": []}
        
        with self._get_session() as session:
            # Create indexes
            for label, properties in KnowledgeGraphSchema.INDEXES.items():
                for property_name in properties:
                    try:
                        query = f"CREATE INDEX {label.lower()}_{property_name}_index IF NOT EXISTS FOR (n:{label}) ON (n.{property_name})"
                        session.run(query)
                        results["indexes"].append(f"{label}.{property_name}")
                        logger.info(f"Created index: {label}.{property_name}")
                    except Exception as e:
                        error_msg = f"Failed to create index {label}.{property_name}: {e}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)
            
            # Create constraints
            for label, properties in KnowledgeGraphSchema.CONSTRAINTS.items():
                for property_name in properties:
                    try:
                        query = f"CREATE CONSTRAINT {label.lower()}_{property_name}_unique IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
                        session.run(query)
                        results["constraints"].append(f"{label}.{property_name}")
                        logger.info(f"Created constraint: {label}.{property_name}")
                    except Exception as e:
                        error_msg = f"Failed to create constraint {label}.{property_name}: {e}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)
        
        return results
    
    def create_node(self, node: BaseNode) -> bool:
        """Create a single node in the knowledge graph."""
        try:
            with self._get_session() as session:
                label = node.__class__.get_label()
                properties = node.to_dict()
                
                # Build CREATE query
                query = f"CREATE (n:{label} $properties) RETURN n"
                result = session.run(query, properties=properties)
                
                if result.single():
                    logger.info(f"Created {label} node: {node.id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to create {node.__class__.__name__} node: {e}")
            return False
    
    def create_nodes_batch(self, nodes: List[BaseNode]) -> Dict[str, Any]:
        """Create multiple nodes in a batch operation."""
        results = {"created": 0, "failed": 0, "errors": []}
        
        with self._get_session() as session:
            for node in nodes:
                try:
                    label = node.__class__.get_label()
                    properties = node.to_dict()
                    
                    query = f"CREATE (n:{label} $properties) RETURN n"
                    result = session.run(query, properties=properties)
                    
                    if result.single():
                        results["created"] += 1
                    else:
                        results["failed"] += 1
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"{node.__class__.__name__} {node.id}: {e}")
                    logger.error(f"Failed to create {node.__class__.__name__} node: {e}")
        
        return results
    
    def create_relationship(self, relationship: Relationship) -> bool:
        """Create a relationship between two nodes."""
        try:
            with self._get_session() as session:
                properties = relationship.to_dict()
                
                query = """
                MATCH (a), (b)
                WHERE a.id = $source_id AND b.id = $target_id
                CREATE (a)-[r:%s $properties]->(b)
                RETURN r
                """ % relationship.relationship_type
                
                result = session.run(query, **properties)
                
                if result.single():
                    logger.info(f"Created relationship: {relationship.source_id} -> {relationship.target_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False
    
    def get_node_by_id(self, node_id: str, label: str = None) -> Optional[Dict[str, Any]]:
        """Get a node by its ID."""
        try:
            with self._get_session() as session:
                if label:
                    query = f"MATCH (n:{label} {{id: $node_id}}) RETURN n"
                else:
                    query = "MATCH (n {id: $node_id}) RETURN n"
                
                result = session.run(query, node_id=node_id)
                record = result.single()
                
                if record:
                    return dict(record["n"])
                return None
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    def get_indicators_by_type(self, indicator_type: IndicatorType, limit: int = 100) -> List[Dict[str, Any]]:
        """Get indicators by type."""
        try:
            with self._get_session() as session:
                query = """
                MATCH (i:Indicator {indicator_type: $indicator_type})
                RETURN i
                ORDER BY i.created_at DESC
                LIMIT $limit
                """
                
                result = session.run(query, indicator_type=indicator_type.value, limit=limit)
                return [dict(record["i"]) for record in result]
        except Exception as e:
            logger.error(f"Failed to get indicators by type {indicator_type}: {e}")
            return []
    
    def get_document_indicators(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all indicators extracted from a specific document."""
        try:
            with self._get_session() as session:
                query = """
                MATCH (d:Document {id: $document_id})-[:CONTAINS]->(i:Indicator)
                RETURN i, d
                ORDER BY i.page_number, i.created_at
                """
                
                result = session.run(query, document_id=document_id)
                return [{"indicator": dict(record["i"]), "document": dict(record["d"])} for record in result]
        except Exception as e:
            logger.error(f"Failed to get document indicators: {e}")
            return []
    
    def get_related_indicators(self, indicator_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get indicators related to a specific indicator through various relationships."""
        try:
            with self._get_session() as session:
                query = """
                MATCH (i:Indicator {id: $indicator_id})-[r*1..$max_depth]-(related:Indicator)
                RETURN DISTINCT related, r
                ORDER BY related.confidence_score DESC
                """
                
                result = session.run(query, indicator_id=indicator_id, max_depth=max_depth)
                return [{"indicator": dict(record["related"]), "relationships": record["r"]} for record in result]
        except Exception as e:
            logger.error(f"Failed to get related indicators: {e}")
            return []
    
    def get_campaign_indicators(self, campaign_id: str) -> List[Dict[str, Any]]:
        """Get all indicators associated with a campaign."""
        try:
            with self._get_session() as session:
                query = """
                MATCH (c:Campaign {id: $campaign_id})-[:USES_INDICATOR]->(i:Indicator)
                RETURN i, c
                ORDER BY i.threat_level DESC, i.confidence_score DESC
                """
                
                result = session.run(query, campaign_id=campaign_id)
                return [{"indicator": dict(record["i"]), "campaign": dict(record["c"])} for record in result]
        except Exception as e:
            logger.error(f"Failed to get campaign indicators: {e}")
            return []
    
    def find_co_occurring_indicators(self, indicator_value: str, 
                                   min_co_occurrences: int = 2) -> List[Dict[str, Any]]:
        """Find indicators that frequently co-occur with the given indicator."""
        try:
            with self._get_session() as session:
                query = """
                MATCH (i1:Indicator {normalized_value: $indicator_value})-[:CO_OCCURS_WITH]-(i2:Indicator)
                WHERE i2.normalized_value <> $indicator_value
                WITH i2, count(*) as co_occurrence_count
                WHERE co_occurrence_count >= $min_co_occurrences
                RETURN i2, co_occurrence_count
                ORDER BY co_occurrence_count DESC
                """
                
                result = session.run(query, 
                                   indicator_value=indicator_value, 
                                   min_co_occurrences=min_co_occurrences)
                return [{"indicator": dict(record["i2"]), "co_occurrence_count": record["co_occurrence_count"]} 
                       for record in result]
        except Exception as e:
            logger.error(f"Failed to find co-occurring indicators: {e}")
            return []
    
    def get_threat_actor_campaigns(self, threat_actor_id: str) -> List[Dict[str, Any]]:
        """Get all campaigns attributed to a threat actor."""
        try:
            with self._get_session() as session:
                query = """
                MATCH (ta:ThreatActor {id: $threat_actor_id})-[:OPERATES]->(c:Campaign)
                RETURN c, ta
                ORDER BY c.created_at DESC
                """
                
                result = session.run(query, threat_actor_id=threat_actor_id)
                return [{"campaign": dict(record["c"]), "threat_actor": dict(record["ta"])} for record in result]
        except Exception as e:
            logger.error(f"Failed to get threat actor campaigns: {e}")
            return []
    
    def search_indicators(self, search_term: str, indicator_types: List[IndicatorType] = None,
                         limit: int = 50) -> List[Dict[str, Any]]:
        """Search for indicators by value or normalized value."""
        try:
            with self._get_session() as session:
                if indicator_types:
                    type_filters = [f"i.indicator_type = '{t.value}'" for t in indicator_types]
                    type_filter = " OR ".join(type_filters)
                    where_clause = f"WHERE ({type_filter}) AND (i.value CONTAINS $search_term OR i.normalized_value CONTAINS $search_term)"
                else:
                    where_clause = "WHERE i.value CONTAINS $search_term OR i.normalized_value CONTAINS $search_term"
                
                query = f"""
                MATCH (i:Indicator)
                {where_clause}
                RETURN i
                ORDER BY i.confidence_score DESC, i.created_at DESC
                LIMIT $limit
                """
                
                result = session.run(query, search_term=search_term, limit=limit)
                return [dict(record["i"]) for record in result]
        except Exception as e:
            logger.error(f"Failed to search indicators: {e}")
            return []
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        try:
            with self._get_session() as session:
                stats = {}
                
                # Node counts by label
                for label in KnowledgeGraphSchema.NODE_LABELS:
                    query = f"MATCH (n:{label}) RETURN count(n) as count"
                    result = session.run(query)
                    stats[f"{label.lower()}_count"] = result.single()["count"]
                
                # Relationship counts
                for rel_type in KnowledgeGraphSchema.RELATIONSHIP_TYPES:
                    query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
                    result = session.run(query)
                    stats[f"{rel_type.lower()}_count"] = result.single()["count"]
                
                # Top indicator types
                query = """
                MATCH (i:Indicator)
                RETURN i.indicator_type, count(i) as count
                ORDER BY count DESC
                LIMIT 10
                """
                result = session.run(query)
                stats["top_indicator_types"] = [{"type": record["i.indicator_type"], "count": record["count"]} 
                                              for record in result]
                
                # Recent activity
                query = """
                MATCH (n)
                WHERE n.created_at >= datetime() - duration('P7D')
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
                """
                result = session.run(query)
                stats["recent_activity"] = [{"label": record["label"], "count": record["count"]} 
                                          for record in result]
                
                return stats
        except Exception as e:
            logger.error(f"Failed to get knowledge graph stats: {e}")
            return {}
    
    def ingest_document_data(self, document_manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest data from document processing pipeline into the knowledge graph.
        
        Args:
            document_manifest: Manifest from PDFExtractor.process()
        
        Returns:
            Dictionary with ingestion results
        """
        results = {
            "document_created": False,
            "indicators_created": 0,
            "chunks_created": 0,
            "relationships_created": 0,
            "errors": []
        }
        
        try:
            # Create document node
            doc_data = document_manifest
            pdf_path = doc_data.get("pdf", "")
            filename = pdf_path.split("/")[-1] if pdf_path else "unknown.pdf"
            
            # Calculate file hash (you might want to implement this)
            file_hash = f"hash_{filename}_{datetime.now().timestamp()}"
            
            document = create_document_node(
                filename=filename,
                file_path=pdf_path,
                file_hash=file_hash,
                file_size=0,  # You might want to get actual file size
                page_count=len(doc_data.get("text", {}).get("pages", [])),
                processing_status="completed",
                extracted_text_length=0,  # Calculate from text data
                chunk_count=0,  # Calculate from chunks
                table_count=len(doc_data.get("tables", {}).get("pages", [])),
                image_count=len(doc_data.get("images", {}).get("pages", []))
            )
            
            if self.create_node(document):
                results["document_created"] = True
                logger.info(f"Created document node: {document.id}")
            else:
                results["errors"].append("Failed to create document node")
                return results
            
            # Process indicators
            indicators_data = doc_data.get("indicators", {})
            if indicators_data and "summary" in indicators_data:
                all_indicators = indicators_data.get("summary", {}).get("all_indicators_file")
                if all_indicators:
                    # Load indicators from file
                    try:
                        with open(all_indicators, 'r') as f:
                            indicators = json.load(f)
                        
                        created_indicators = []
                        for indicator_type, indicator_list in indicators.items():
                            for indicator_data in indicator_list:
                                try:
                                    # Map indicator type
                                    type_mapping = {
                                        'domains': IndicatorType.DOMAIN,
                                        'urls': IndicatorType.URL,
                                        'ip_addresses': IndicatorType.IP_ADDRESS,
                                        'emails': IndicatorType.EMAIL,
                                        'phone_numbers': IndicatorType.PHONE,
                                        'social_media': IndicatorType.SOCIAL_MEDIA,
                                        'tracking_ids': IndicatorType.TRACKING_ID
                                    }
                                    
                                    neo4j_type = type_mapping.get(indicator_type, IndicatorType.DOMAIN)
                                    
                                    indicator = create_indicator_node(
                                        value=indicator_data.get("value", ""),
                                        normalized_value=indicator_data.get("normalized", ""),
                                        indicator_type=neo4j_type,
                                        context=indicator_data.get("context", ""),
                                        confidence_score=0.8,  # Default confidence
                                        source=filename
                                    )
                                    
                                    created_indicators.append(indicator)
                                    
                                except Exception as e:
                                    results["errors"].append(f"Failed to create indicator: {e}")
                        
                        # Batch create indicators
                        if created_indicators:
                            batch_results = self.create_nodes_batch(created_indicators)
                            results["indicators_created"] = batch_results["created"]
                            results["errors"].extend(batch_results["errors"])
                            
                            # Create document-indicator relationships
                            for indicator in created_indicators:
                                if batch_results["created"] > 0:  # Only if indicator was created
                                    rel = DocumentIndicatorRelationship(
                                        source_id=document.id,
                                        target_id=indicator.id,
                                        relationship_type=RelationshipType.CONTAINS,
                                        extraction_context=indicator.context,
                                        extraction_confidence=indicator.confidence_score
                                    )
                                    
                                    if self.create_relationship(rel):
                                        results["relationships_created"] += 1
                    
                    except Exception as e:
                        results["errors"].append(f"Failed to load indicators: {e}")
            
            # Process chunks if available
            unified_collection = doc_data.get("unified_collection", {})
            if unified_collection and "chunks" in unified_collection:
                chunks_data = unified_collection["chunks"]
                created_chunks = []
                
                for chunk_data in chunks_data:
                    try:
                        chunk = create_chunk_node(
                            content=chunk_data.get("content", ""),
                            chunk_index=chunk_data.get("chunk_index", 0),
                            content_type=chunk_data.get("content_type", "text"),
                            page_number=chunk_data.get("page", 0),
                            chunk_size=chunk_data.get("chunk_size", 0),
                            source=filename
                        )
                        created_chunks.append(chunk)
                    except Exception as e:
                        results["errors"].append(f"Failed to create chunk: {e}")
                
                # Batch create chunks
                if created_chunks:
                    batch_results = self.create_nodes_batch(created_chunks)
                    results["chunks_created"] = batch_results["created"]
                    results["errors"].extend(batch_results["errors"])
                    
                    # Create document-chunk relationships
                    for chunk in created_chunks:
                        if batch_results["created"] > 0:
                            rel = Relationship(
                                source_id=document.id,
                                target_id=chunk.id,
                                relationship_type=RelationshipType.CONTAINS
                            )
                            
                            if self.create_relationship(rel):
                                results["relationships_created"] += 1
        
        except Exception as e:
            results["errors"].append(f"Failed to ingest document data: {e}")
            logger.error(f"Document ingestion failed: {e}")
        
        return results
    
    def execute_custom_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query."""
        try:
            with self._get_session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to execute custom query: {e}")
            return []
