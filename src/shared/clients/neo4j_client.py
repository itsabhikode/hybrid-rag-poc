from typing import Dict, Any

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: Neo4j driver not available. Knowledge graph features will be disabled.")


class Neo4jKnowledgeGraphClient:
    """Simple Neo4j client for knowledge graph operations with lazy initialization."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", password: str = "testpass"):
        """Initialize Neo4j client with lazy connection."""
        self.uri = uri
        self.username = username
        self.password = password
        self._driver = None
        self._connected = False
    
    @property
    def driver(self):
        """Lazy initialization of Neo4j driver."""
        if not NEO4J_AVAILABLE:
            return None
            
        if self._driver is None and not self._connected:
            try:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
                # Test connection
                with self._driver.session() as session:
                    session.run("RETURN 1")
                print(f"Connected to Neo4j at {self.uri}")
                self._connected = True
            except Exception as e:
                print(f"Failed to connect to Neo4j: {e}")
                self._driver = None
                self._connected = False
                
        return self._driver
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
    
    def setup_schema(self):
        """Set up the knowledge graph schema with indexes and constraints."""
        if not self.driver:
            return {"error": "Neo4j not available"}
        
        results = {"indexes": [], "constraints": [], "errors": []}
        
        with self.driver.session() as session:
            # Create indexes
            indexes = [
                ("Document", "filename"),
                ("Document", "file_hash"),
                ("Indicator", "value"),
                ("Indicator", "normalized_value"),
                ("Indicator", "indicator_type"),
                ("Campaign", "name"),
                ("ThreatActor", "name")
            ]
            
            for label, property_name in indexes:
                try:
                    query = f"CREATE INDEX {label.lower()}_{property_name}_index IF NOT EXISTS FOR (n:{label}) ON (n.{property_name})"
                    session.run(query)
                    results["indexes"].append(f"{label}.{property_name}")
                except Exception as e:
                    results["errors"].append(f"Failed to create index {label}.{property_name}: {e}")
            
            # Create constraints
            constraints = [
                ("Document", "file_hash"),
                ("Indicator", "normalized_value"),
                ("Campaign", "name"),
                ("ThreatActor", "name")
            ]
            
            for label, property_name in constraints:
                try:
                    query = f"CREATE CONSTRAINT {label.lower()}_{property_name}_unique IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
                    session.run(query)
                    results["constraints"].append(f"{label}.{property_name}")
                except Exception as e:
                    results["errors"].append(f"Failed to create constraint {label}.{property_name}: {e}")
        
        return results
    
    def create_document_node(self, document_data: Dict[str, Any]) -> str:
        """Create a document node in the knowledge graph."""
        if not self.driver:
            return None
        
        with self.driver.session() as session:
            query = """
            CREATE (d:Document {
                id: $id,
                filename: $filename,
                file_path: $file_path,
                file_hash: $file_hash,
                file_size: $file_size,
                page_count: $page_count,
                language: $language,
                document_type: $document_type,
                processing_status: $processing_status,
                extracted_text_length: $extracted_text_length,
                chunk_count: $chunk_count,
                table_count: $table_count,
                image_count: $image_count,
                created_at: datetime(),
                updated_at: datetime(),
                confidence_score: $confidence_score,
                source: $source
            })
            RETURN d.id as id
            """
            
            result = session.run(query, **document_data)
            record = result.single()
            return record["id"] if record else None
    
    def create_indicator_node(self, indicator_data: Dict[str, Any]) -> str:
        """Create an indicator node in the knowledge graph."""
        if not self.driver:
            return None
        
        with self.driver.session() as session:
            query = """
            CREATE (i:Indicator {
                id: $id,
                value: $value,
                normalized_value: $normalized_value,
                indicator_type: $indicator_type,
                context: $context,
                page_number: $page_number,
                chunk_id: $chunk_id,
                extraction_method: $extraction_method,
                threat_level: $threat_level,
                is_verified: $is_verified,
                verification_date: $verification_date,
                verification_source: $verification_source,
                created_at: datetime(),
                updated_at: datetime(),
                confidence_score: $confidence_score,
                source: $source
            })
            RETURN i.id as id
            """
            
            result = session.run(query, **indicator_data)
            record = result.single()
            return record["id"] if record else None
    
    def create_chunk_node(self, chunk_data: Dict[str, Any]) -> str:
        """Create a chunk node in the knowledge graph."""
        if not self.driver:
            return None
        
        with self.driver.session() as session:
            query = """
            CREATE (c:Chunk {
                id: $id,
                content: $content,
                chunk_index: $chunk_index,
                content_type: $content_type,
                page_number: $page_number,
                chunk_size: $chunk_size,
                embedding_id: $embedding_id,
                created_at: datetime(),
                updated_at: datetime(),
                confidence_score: $confidence_score,
                source: $source
            })
            RETURN c.id as id
            """
            
            result = session.run(query, **chunk_data)
            record = result.single()
            return record["id"] if record else None
    
    def create_relationship(self, source_id: str, target_id: str, 
                          relationship_type: str, properties: Dict[str, Any] = None):
        """Create a relationship between two nodes."""
        if not self.driver:
            return False
        
        with self.driver.session() as session:
            if properties:
                props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                query = f"""
                MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                CREATE (a)-[r:{relationship_type} {{{props_str}}}]->(b)
                RETURN r
                """
                params = {"source_id": source_id, "target_id": target_id, **properties}
            else:
                query = f"""
                MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                CREATE (a)-[r:{relationship_type}]->(b)
                RETURN r
                """
                params = {"source_id": source_id, "target_id": target_id}
            
            result = session.run(query, **params)
            return result.single() is not None
    
    def discover_co_occurrence_relationships(self, min_co_occurrences: int = 2):
        """Discover co-occurrence relationships between indicators."""
        if not self.driver:
            return {"relationships_created": 0}
        
        with self.driver.session() as session:
            query = """
            MATCH (d:Document)-[:CONTAINS]->(i1:Indicator)
            MATCH (d)-[:CONTAINS]->(i2:Indicator)
            WHERE i1.id < i2.id
            WITH i1, i2, count(d) as co_occurrence_count
            WHERE co_occurrence_count >= $min_co_occurrences
            MERGE (i1)-[r:CO_OCCURS_WITH {
                co_occurrence_count: co_occurrence_count,
                confidence_score: CASE 
                    WHEN co_occurrence_count >= 5 THEN 0.9
                    WHEN co_occurrence_count >= 3 THEN 0.7
                    ELSE 0.5
                END,
                created_at: datetime()
            }]-(i2)
            RETURN count(r) as relationships_created
            """
            
            result = session.run(query, min_co_occurrences=min_co_occurrences)
            record = result.single()
            return {"relationships_created": record["relationships_created"] if record else 0}
