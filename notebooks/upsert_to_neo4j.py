#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Upsert Script (fixed)

- Ensures every node stores a stable `id` property.
- Relationship MATCH strictly by {id: ...} for both endpoints.
- Upserts are idempotent; relationship properties are merged using coalesce().
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("❌ Neo4j driver not available. Install with: pip install neo4j")
    sys.exit(1)


class Neo4jUpserter:
    """Handle upserting knowledge graph data to Neo4j."""
    def __init__(self, uri: str = "bolt://localhost:7687",
                 username: str = "neo4j", password: str = "testpass"):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None

    def connect(self):
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"✅ Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        if self.driver:
            self.driver.close()

    def setup_schema(self):
        """Create indexes/constraints for performance and uniqueness."""
        if not self.driver:
            return False

        print("🔧 Setting up Neo4j schema...")
        with self.driver.session() as session:
            indexes = [
                # General
                ("Document", "id"),
                ("Indicator", "id"),
                ("ThreatActor", "id"),
                ("Campaign", "id"),
                # Existing useful props
                ("Document", "filename"),
                ("Document", "file_hash"),
                ("Indicator", "value"),
                ("Indicator", "normalized_value"),
                ("Indicator", "indicator_type"),
                ("ThreatActor", "name"),
                ("ThreatActor", "normalized_name"),
                ("Campaign", "name"),
                ("Campaign", "normalized_name"),
            ]
            for label, prop in indexes:
                try:
                    session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")
                    print(f"  ✅ Index on {label}.{prop}")
                except Exception as e:
                    print(f"  ⚠️ Index on {label}.{prop}: {e}")

            constraints = [
                # Keep prior uniqueness
                ("Document", "file_hash", "UNIQUE"),
                ("Indicator", "value", "UNIQUE"),
                ("ThreatActor", "normalized_name", "UNIQUE"),
                ("Campaign", "normalized_name", "UNIQUE"),
                # Add uniqueness on id per label (assuming ids are globally unique per label)
                ("Document", "id", "UNIQUE"),
                ("Indicator", "id", "UNIQUE"),
                ("ThreatActor", "id", "UNIQUE"),
                ("Campaign", "id", "UNIQUE"),
            ]
            for label, prop, kind in constraints:
                try:
                    if kind == "UNIQUE":
                        session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
                        print(f"  ✅ Unique on {label}.{prop}")
                except Exception as e:
                    print(f"  ⚠️ Constraint on {label}.{prop}: {e}")

        return True

    def clear_database(self):
        if not self.driver:
            return False
        print("🗑️  Clearing database...")
        with self.driver.session() as session:
            session.run("MATCH ()-[r]->() DELETE r")
            print("  ✅ Deleted all relationships")
            session.run("MATCH (n) DELETE n")
            print("  ✅ Deleted all nodes")
        return True

    def upsert_document(self, document_data: Dict[str, Any]) -> str:
        """Upsert a Document node (MERGE by file_hash; store id)."""
        mapped = {
            "id": document_data.get("id"),
            "file_hash": document_data.get("file_hash"),
            "filename": document_data.get("filename"),
            "file_path": document_data.get("file_path"),
            "file_size": document_data.get("file_size"),
            "created_at": document_data.get("extraction_timestamp", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "extraction_timestamp": document_data.get("extraction_timestamp"),
            "total_pages": document_data.get("page_count", 0),
            "total_chunks": document_data.get("chunk_count", 0),
            "has_indicators": True,
            "has_threat_actors": True,
            "has_campaigns": True,
            "document_type": document_data.get("document_type", "pdf"),
            "processing_status": document_data.get("processing_status", "completed"),
            "language": document_data.get("language", "en"),
            "confidence_score": document_data.get("confidence_score", 1.0)
        }
        query = """
        MERGE (d:Document {file_hash: $file_hash})
        SET d.id = $id,
            d.filename = $filename,
            d.file_path = $file_path,
            d.file_size = $file_size,
            d.created_at = $created_at,
            d.updated_at = $updated_at,
            d.extraction_timestamp = $extraction_timestamp,
            d.total_pages = $total_pages,
            d.total_chunks = $total_chunks,
            d.has_indicators = $has_indicators,
            d.has_threat_actors = $has_threat_actors,
            d.has_campaigns = $has_campaigns,
            d.document_type = $document_type,
            d.processing_status = $processing_status,
            d.language = $language,
            d.confidence_score = $confidence_score
        RETURN d.file_hash AS file_hash
        """
        with self.driver.session() as session:
            result = session.run(query, **mapped)
            return result.single()["file_hash"]

    def upsert_indicator(self, indicator_data: Dict[str, Any], document_hash: str) -> str:
        """Upsert an Indicator and create MENTIONED_IN (Indicator)->(Document)."""
        mapped = {
            "id": indicator_data.get("id"),
            "value": indicator_data.get("value"),
            "normalized_value": indicator_data.get("normalized_value"),
            "indicator_type": indicator_data.get("indicator_type"),
            "confidence_score": indicator_data.get("confidence_score", 0.8),
            "extraction_context": indicator_data.get("context", ""),
            "page_number": indicator_data.get("page_number"),
            "chunk_index": indicator_data.get("chunk_id"),
            "created_at": indicator_data.get("extraction_timestamp", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "extraction_method": indicator_data.get("extraction_method", "regex"),
            "source_document": indicator_data.get("source_document")
        }
        upsert_q = """
        MERGE (i:Indicator {value: $value})
        SET i.id = $id,
            i.normalized_value = $normalized_value,
            i.indicator_type = $indicator_type,
            i.confidence_score = $confidence_score,
            i.extraction_context = $extraction_context,
            i.page_number = $page_number,
            i.chunk_index = $chunk_index,
            i.created_at = $created_at,
            i.updated_at = $updated_at,
            i.extraction_method = $extraction_method,
            i.source_document = $source_document
        RETURN i.value AS value
        """
        rel_q = """
        MATCH (d:Document {file_hash: $document_hash})
        MATCH (i:Indicator {value: $indicator_value})
        MERGE (i)-[r:MENTIONED_IN]->(d)
        SET r.confidence_score      = coalesce($confidence_score, r.confidence_score, 0.8),
            r.extraction_context    = coalesce($extraction_context, r.extraction_context),
            r.page_number           = coalesce($page_number, r.page_number),
            r.chunk_index           = coalesce($chunk_index, r.chunk_index),
            r.created_at            = coalesce($created_at, r.created_at, datetime())
        RETURN r
        """
        with self.driver.session() as session:
            res = session.run(upsert_q, **mapped)
            indicator_value = res.single()["value"]
            session.run(
                rel_q,
                document_hash=document_hash,
                indicator_value=indicator_value,
                confidence_score=mapped["confidence_score"],
                extraction_context=mapped["extraction_context"],
                page_number=mapped["page_number"],
                chunk_index=mapped["chunk_index"],
                created_at=mapped["created_at"],
            )
            return indicator_value

    def upsert_threat_actor(self, threat_actor_data: Dict[str, Any], document_hash: str) -> str:
        """Upsert a ThreatActor and create MENTIONED_IN (ThreatActor)->(Document)."""
        mapped = {
            "id": threat_actor_data.get("id"),  # <-- ensure id present
            "name": threat_actor_data.get("name"),
            "normalized_name": threat_actor_data.get("normalized_name"),
            "entity_type": threat_actor_data.get("entity_type"),
            "confidence_score": threat_actor_data.get("confidence_score", 1.0),
            "extraction_context": threat_actor_data.get("context", ""),
            "page_number": threat_actor_data.get("page_number"),
            "chunk_index": None,
            "created_at": threat_actor_data.get("extraction_timestamp", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "extraction_method": threat_actor_data.get("extraction_method", "spacy_ner"),
            "source_document": threat_actor_data.get("source_document"),
        }
        upsert_q = """
        MERGE (ta:ThreatActor {normalized_name: $normalized_name})
        SET ta.id = $id,
            ta.name = $name,
            ta.entity_type = $entity_type,
            ta.confidence_score = $confidence_score,
            ta.extraction_context = $extraction_context,
            ta.page_number = $page_number,
            ta.chunk_index = $chunk_index,
            ta.created_at = $created_at,
            ta.updated_at = $updated_at,
            ta.extraction_method = $extraction_method,
            ta.source_document = $source_document
        RETURN ta.normalized_name AS normalized_name
        """
        rel_q = """
        MATCH (d:Document {file_hash: $document_hash})
        MATCH (ta:ThreatActor {normalized_name: $threat_actor_name})
        MERGE (ta)-[r:MENTIONED_IN]->(d)
        SET r.confidence_score      = coalesce($confidence_score, r.confidence_score, 1.0),
            r.extraction_context    = coalesce($extraction_context, r.extraction_context),
            r.page_number           = coalesce($page_number, r.page_number),
            r.chunk_index           = coalesce($chunk_index, r.chunk_index),
            r.created_at            = coalesce($created_at, r.created_at, datetime())
        RETURN r
        """
        with self.driver.session() as session:
            res = session.run(upsert_q, **mapped)
            name = res.single()["normalized_name"]
            session.run(
                rel_q,
                document_hash=document_hash,
                threat_actor_name=name,
                confidence_score=mapped["confidence_score"],
                extraction_context=mapped["extraction_context"],
                page_number=mapped["page_number"],
                chunk_index=mapped["chunk_index"],
                created_at=mapped["created_at"],
            )
            return name

    def upsert_campaign(self, campaign_data: Dict[str, Any], document_hash: str) -> str:
        """Upsert a Campaign and create MENTIONED_IN (Campaign)->(Document)."""
        mapped = {
            "id": campaign_data.get("id"),  # <-- ensure id present
            "name": campaign_data.get("name"),
            "normalized_name": campaign_data.get("normalized_name"),
            "campaign_type": campaign_data.get("campaign_type", "cyber_operation"),
            "confidence_score": campaign_data.get("confidence_score", 0.7),
            "extraction_context": "",
            "page_number": None,
            "chunk_index": None,
            "created_at": campaign_data.get("extraction_timestamp", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "extraction_method": campaign_data.get("extraction_method", "pattern_matching"),
            "source_document": campaign_data.get("source_document"),
        }
        upsert_q = """
        MERGE (c:Campaign {normalized_name: $normalized_name})
        SET c.id = $id,
            c.name = $name,
            c.campaign_type = $campaign_type,
            c.confidence_score = $confidence_score,
            c.extraction_context = $extraction_context,
            c.page_number = $page_number,
            c.chunk_index = $chunk_index,
            c.created_at = $created_at,
            c.updated_at = $updated_at,
            c.extraction_method = $extraction_method,
            c.source_document = $source_document
        RETURN c.normalized_name AS normalized_name
        """
        rel_q = """
        MATCH (d:Document {file_hash: $document_hash})
        MATCH (c:Campaign {normalized_name: $campaign_name})
        MERGE (c)-[r:MENTIONED_IN]->(d)
        SET r.confidence_score      = coalesce($confidence_score, r.confidence_score, 0.7),
            r.extraction_context    = coalesce($extraction_context, r.extraction_context),
            r.page_number           = coalesce($page_number, r.page_number),
            r.chunk_index           = coalesce($chunk_index, r.chunk_index),
            r.created_at            = coalesce($created_at, r.created_at, datetime())
        RETURN r
        """
        with self.driver.session() as session:
            res = session.run(upsert_q, **mapped)
            name = res.single()["normalized_name"]
            session.run(
                rel_q,
                document_hash=document_hash,
                campaign_name=name,
                confidence_score=mapped["confidence_score"],
                extraction_context=mapped["extraction_context"],
                page_number=mapped["page_number"],
                chunk_index=mapped["chunk_index"],
                created_at=mapped["created_at"],
            )
            return name

    def upsert_relationships(self, relationships: List[Dict[str, Any]]):
        """Upsert relationships. Endpoints are matched strictly by node `id`."""
        if not relationships:
            return
        print(f"🔗 Upserting {len(relationships)} relationships...")

        successful_rels = 0
        failed_rels = 0
        missing_nodes = set()

        with self.driver.session() as session:
            for rel in relationships:
                try:
                    rel_type = rel["relationship_type"]
                    from_id = rel["from_value"]
                    to_id = rel["to_value"]
                    props = rel.get("properties", {}) or {}

                    # First check if both nodes exist
                    check_query = """
                    MATCH (from {id: $from_id})
                    MATCH (to {id: $to_id})
                    RETURN from.id as from_exists, to.id as to_exists
                    """
                    result = session.run(check_query, from_id=from_id, to_id=to_id)
                    record = result.single()
                    
                    if not record:
                        missing_nodes.add(f"{from_id} or {to_id}")
                        failed_rels += 1
                        print(f"  ⚠️  Skipping relationship {rel_type}: nodes not found ({from_id} -> {to_id})")
                        continue

                    if rel_type == "MENTIONED_IN":
                        q = """
                        MATCH (from {id: $from_id})
                        MATCH (to   {id: $to_id})
                        MERGE (from)-[r:MENTIONED_IN]->(to)
                        SET r.extraction_context    = coalesce($extraction_context, r.extraction_context),
                            r.extraction_confidence = coalesce($extraction_confidence, r.extraction_confidence, 0.8),
                            r.page_number           = coalesce($page_number, r.page_number),
                            r.extraction_timestamp  = coalesce($extraction_timestamp, r.extraction_timestamp, datetime())
                        RETURN r
                        """
                        session.run(
                            q,
                            from_id=from_id,
                            to_id=to_id,
                            extraction_context=props.get("extraction_context"),
                            extraction_confidence=props.get("extraction_confidence", 0.8),
                            page_number=props.get("page_number"),
                            extraction_timestamp=props.get("extraction_timestamp", datetime.now().isoformat()),
                        )

                    elif rel_type == "RELATED_TO":
                        q = """
                        MATCH (from {id: $from_id})
                        MATCH (to   {id: $to_id})
                        MERGE (from)-[r:RELATED_TO]->(to)
                        SET r.confidence_score  = coalesce($confidence_score, r.confidence_score, 0.8),
                            r.extraction_context= coalesce($extraction_context, r.extraction_context),
                            r.created_at        = coalesce($created_at, r.created_at, datetime())
                        RETURN r
                        """
                        session.run(
                            q,
                            from_id=from_id,
                            to_id=to_id,
                            confidence_score=props.get("extraction_confidence", 0.8),
                            extraction_context=props.get("extraction_context"),
                            created_at=props.get("extraction_timestamp", datetime.now().isoformat()),
                        )

                    elif rel_type == "PART_OF_CAMPAIGN":
                        q = """
                        MATCH (from {id: $from_id})
                        MATCH (to   {id: $to_id})
                        MERGE (from)-[r:PART_OF_CAMPAIGN]->(to)
                        SET r.confidence_score  = coalesce($confidence_score, r.confidence_score, 0.8),
                            r.extraction_context= coalesce($extraction_context, r.extraction_context),
                            r.created_at        = coalesce($created_at, r.created_at, datetime())
                        RETURN r
                        """
                        session.run(
                            q,
                            from_id=from_id,
                            to_id=to_id,
                            confidence_score=props.get("extraction_confidence", 0.8),
                            extraction_context=props.get("extraction_context"),
                            created_at=props.get("extraction_timestamp", datetime.now().isoformat()),
                        )

                    successful_rels += 1

                except Exception as e:
                    failed_rels += 1
                    print(f"  ⚠️  Error upserting relationship {rel.get('relationship_type', 'unknown')}: {e}")
                    continue

        print(f"  ✅ Successfully created {successful_rels} relationships")
        if failed_rels > 0:
            print(f"  ⚠️  Failed to create {failed_rels} relationships")
            if missing_nodes:
                print(f"  📝 Missing nodes: {len(missing_nodes)} unique node pairs")

    def upsert_neo4j_data(self, neo4j_data: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        """Upsert a complete graph bundle from your JSON schema."""
        if dry_run:
            print("🔍 DRY RUN - No data will be actually upserted")
            return self._analyze_neo4j_data(neo4j_data)

        results = {"documents": 0, "indicators": 0, "threat_actors": 0, "campaigns": 0, "relationships": 0, "errors": []}
        try:
            # Document (assume 1 per file)
            documents = neo4j_data["nodes"]["documents"]
            if not documents:
                raise ValueError("No documents found in data")

            document_node = documents[0]["properties"].copy()
            document_node["id"] = documents[0]["id"]  # ensure id in node
            document_hash = self.upsert_document(document_node)
            results["documents"] = 1
            print(f"✅ Upserted document: {document_node.get('filename')}")

            # Indicators
            for indicator in neo4j_data["nodes"]["indicators"]:
                try:
                    props = indicator["properties"].copy()
                    props["id"] = indicator["id"]  # ensure id in node
                    self.upsert_indicator(props, document_hash)
                    results["indicators"] += 1
                except Exception as e:
                    results["errors"].append(f"Indicator {indicator.get('properties', {}).get('value', 'unknown')}: {e}")

            print(f"✅ Upserted {results['indicators']} indicators")

            # Threat Actors
            for ta in neo4j_data["nodes"]["threat_actors"]:
                try:
                    props = ta["properties"].copy()
                    props["id"] = ta["id"]  # ensure id in node
                    self.upsert_threat_actor(props, document_hash)
                    results["threat_actors"] += 1
                except Exception as e:
                    results["errors"].append(f"Threat actor {ta.get('properties', {}).get('name', 'unknown')}: {e}")

            print(f"✅ Upserted {results['threat_actors']} threat actors")

            # Campaigns
            for camp in neo4j_data["nodes"]["campaigns"]:
                try:
                    props = camp["properties"].copy()
                    props["id"] = camp["id"]  # ensure id in node
                    self.upsert_campaign(props, document_hash)
                    results["campaigns"] += 1
                except Exception as e:
                    results["errors"].append(f"Campaign {camp.get('properties', {}).get('name', 'unknown')}: {e}")

            print(f"✅ Upserted {results['campaigns']} campaigns")

            # Edges → normalize into relationships list
            all_relationships: List[Dict[str, Any]] = []
            for edge_type, edges in neo4j_data.get("edges", {}).items():
                for edge in edges:
                    # Expecting edge 'from' and 'to' to be NODE IDS from your JSON graph
                    edge_props = edge.get("properties", {}) or {}
                    all_relationships.append({
                        "relationship_type": edge_type.upper(),
                        "from_value": edge.get("from", ""),
                        "to_value": edge.get("to", ""),
                        "properties": {
                            "extraction_confidence": edge_props.get("confidence_score", 0.8),
                            "extraction_context": edge_props.get("extraction_context", ""),
                            "extraction_timestamp": edge_props.get("created_at", datetime.now().isoformat()),
                            "page_number": edge_props.get("page_number"),
                        }
                    })

            self.upsert_relationships(all_relationships)
            results["relationships"] = len(all_relationships)
            print(f"✅ Upserted {results['relationships']} relationships")

        except Exception as e:
            results["errors"].append(f"General error: {e}")
            print(f"❌ Error during upsert: {e}")

        return results

    def _analyze_neo4j_data(self, neo4j_data: Dict[str, Any]) -> Dict[str, Any]:
        total_relationships = sum(len(edges) for edges in neo4j_data.get("edges", {}).values())
        documents = neo4j_data["nodes"]["documents"]
        document_info = {}
        if documents:
            doc_props = documents[0]["properties"]
            document_info = {
                "filename": doc_props.get("filename", "unknown"),
                "file_hash": doc_props.get("file_hash", "unknown"),
                "total_pages": doc_props.get("total_pages", doc_props.get("page_count", 0)),
                "total_chunks": doc_props.get("total_chunks", doc_props.get("chunk_count", 0)),
            }

        analysis = {
            "documents": len(neo4j_data["nodes"]["documents"]),
            "indicators": len(neo4j_data["nodes"]["indicators"]),
            "threat_actors": len(neo4j_data["nodes"]["threat_actors"]),
            "campaigns": len(neo4j_data["nodes"]["campaigns"]),
            "relationships": total_relationships,
            "document_info": document_info
        }

        print("📊 Analysis:")
        if document_info:
            print(f"  Document: {analysis['document_info']['filename']}")
        print(f"  Documents: {analysis['documents']}")
        print(f"  Indicators: {analysis['indicators']}")
        print(f"  Threat Actors: {analysis['threat_actors']}")
        print(f"  Campaigns: {analysis['campaigns']}")
        print(f"  Relationships: {analysis['relationships']}")
        return analysis


def find_neo4j_data_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return list(data_dir.glob("neo4j_data_*.json"))

def load_neo4j_data(file_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Upsert knowledge graph data to Neo4j")
    parser.add_argument("--neo4j-data-dir", type=str, default=".", help="Directory containing Neo4j data JSON files")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--neo4j-username", type=str, default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", type=str, default="testpass", help="Neo4j password")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be upserted without actually doing it")
    parser.add_argument("--clear-db", action="store_true", help="Clear the database before upserting")
    parser.add_argument("--file", type=str, help="Specific Neo4j data file to upsert")
    args = parser.parse_args()

    if not NEO4J_AVAILABLE:
        print("❌ Neo4j driver not available. Install with: pip install neo4j")
        return 1

    upserter = Neo4jUpserter(args.neo4j_uri, args.neo4j_username, args.neo4j_password)

    if not args.dry_run:
        if not upserter.connect():
            return 1
        upserter.setup_schema()
        if args.clear_db:
            upserter.clear_database()

    data_dir = Path(args.neo4j_data_dir)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return 1
        neo4j_data = load_neo4j_data(file_path)
        if neo4j_data:
            print(f"📄 Processing: {file_path}")
            results = upserter.upsert_neo4j_data(neo4j_data, args.dry_run)
            if not args.dry_run:
                print("\n📊 Upsert Results:")
                print(f"  Documents: {results['documents']}")
                print(f"  Indicators: {results['indicators']}")
                print(f"  Threat Actors: {results['threat_actors']}")
                print(f"  Campaigns: {results['campaigns']}")
                print(f"  Relationships: {results['relationships']}")
                if results['errors']:
                    print("\n❌ Errors:")
                    for e in results['errors']:
                        print(f"  - {e}")
    else:
        neo4j_files = find_neo4j_data_files(data_dir)
        if not neo4j_files:
            print(f"❌ No Neo4j data files found in {data_dir}")
            print("   Looking for files matching pattern: neo4j_data_*.json")
            return 1

        print(f"📁 Found {len(neo4j_files)} Neo4j data files")
        total = {"documents": 0, "indicators": 0, "threat_actors": 0, "campaigns": 0, "relationships": 0, "errors": []}
        for file_path in neo4j_files:
            print(f"\n📄 Processing: {file_path.name}")
            neo4j_data = load_neo4j_data(file_path)
            if neo4j_data:
                results = upserter.upsert_neo4j_data(neo4j_data, args.dry_run)
                for k in ["documents", "indicators", "threat_actors", "campaigns", "relationships"]:
                    total[k] += results[k]
                total["errors"].extend(results["errors"])

        if not args.dry_run:
            print("\n📊 Total Upsert Results:")
            print(f"  Documents: {total['documents']}")
            print(f"  Indicators: {total['indicators']}")
            print(f"  Threat Actors: {total['threat_actors']}")
            print(f"  Campaigns: {total['campaigns']}")
            print(f"  Relationships: {total['relationships']}")
            if total['errors']:
                print("\n❌ Errors:")
                for e in total['errors']:
                    print(f"  - {e}")

    if not args.dry_run:
        upserter.close()

    print("\n✅ Neo4j upsert completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
