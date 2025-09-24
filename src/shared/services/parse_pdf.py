import json
import re
from pathlib import Path
import pymupdf as fitz  # PyMuPDF
import pymupdf4llm
import hashlib
from datetime import datetime
import uuid
from typing import Dict, Any

# Import from shared modules
import sys
sys.path.append('/Users/akarna/Blackbox POC/src/shared/clients')
sys.path.append('/Users/akarna/Blackbox POC/src/shared/services')

from shared.clients.neo4j_client import Neo4jKnowledgeGraphClient
from shared.clients.qdrant_client import QdrantVectorStore
from shared.services.indicator import IndicatorExtractor
from shared.services.chunk_service import TextChunker
from shared.services.threat_actor_extractor import ThreatActorExtractor
from shared.services.neo4j_data_extractor import Neo4jDataExtractor

class PDFExtractor:
    def __init__(self, pdf_path: Path, out_root: Path, save_blocks: bool = False, extract_indicators: bool = True,
                 enable_chunking: bool = True, chunk_size: int = 1000, chunk_overlap: int = 250,
                 enable_qdrant: bool = False, qdrant_host: str = "localhost", qdrant_port: int = 6333,
                 qdrant_collection: str = "document_chunks", embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 qdrant_api_key: str = None, enable_neo4j: bool = False, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_username: str = "neo4j", neo4j_password: str = "testpass", extract_threat_actors: bool = True):
        self.pdf_path = Path(pdf_path)
        self.out_root = Path(out_root)
        self.save_blocks = save_blocks
        self.extract_indicators = extract_indicators
        self.extract_threat_actors = extract_threat_actors
        self.enable_chunking = enable_chunking
        self.enable_qdrant = enable_qdrant
        self.enable_neo4j = enable_neo4j

        # Directories
        self.pdf_dir = self.ensure_dir(self.out_root / self.pdf_path.stem)
        self.text_dir = self.ensure_dir(self.pdf_dir / "text")
        self.img_dir = self.ensure_dir(self.pdf_dir / "images")
        self.tbl_dir = self.ensure_dir(self.pdf_dir / "tables")
        self.indicators_dir = self.ensure_dir(self.pdf_dir / "indicators")
        self.threat_actors_dir = self.ensure_dir(self.pdf_dir / "threat_actors")
        self.neo4j_dir = self.ensure_dir(self.pdf_dir / "neo4j")
        self.chunks_dir = self.ensure_dir(self.pdf_dir / "chunks")

        # Document object
        self.doc = fitz.open(str(self.pdf_path))
        
        # Store configuration for lazy initialization
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_collection = qdrant_collection
        self.embedding_model = embedding_model
        self.qdrant_api_key = qdrant_api_key
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        
        # Lazy initialization - services will be created when needed
        self._indicator_extractor = None
        self._threat_actor_extractor = None
        self._neo4j_data_extractor = None
        self._text_chunker = None
        self._qdrant_store = None
        self._neo4j_client = None

    # ---------- Utility ----------
    @staticmethod
    def ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def save_json(path: Path, data):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # ---------- Lazy Initialization Properties ----------
    @property
    def indicator_extractor(self):
        """Lazy initialization of indicator extractor."""
        if self._indicator_extractor is None and self.extract_indicators:
            self._indicator_extractor = IndicatorExtractor()
        return self._indicator_extractor
    
    @property
    def threat_actor_extractor(self):
        """Lazy initialization of threat actor extractor."""
        if self._threat_actor_extractor is None and self.extract_threat_actors:
            self._threat_actor_extractor = ThreatActorExtractor()
        return self._threat_actor_extractor
    
    @property
    def neo4j_data_extractor(self):
        """Lazy initialization of Neo4j data extractor."""
        if self._neo4j_data_extractor is None:
            self._neo4j_data_extractor = Neo4jDataExtractor()
        return self._neo4j_data_extractor
    
    @property
    def text_chunker(self):
        """Lazy initialization of text chunker."""
        if self._text_chunker is None and self.enable_chunking:
            self._text_chunker = TextChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return self._text_chunker
    
    @property
    def qdrant_store(self):
        """Lazy initialization of Qdrant vector store."""
        if self._qdrant_store is None and self.enable_qdrant:
            self._qdrant_store = QdrantVectorStore(
                host=self.qdrant_host,
                port=self.qdrant_port,
                collection_name=self.qdrant_collection,
                embedding_model=self.embedding_model,
                api_key=self.qdrant_api_key
            )
        return self._qdrant_store
    
    @property
    def neo4j_client(self):
        """Lazy initialization of Neo4j knowledge graph client."""
        if self._neo4j_client is None and self.enable_neo4j:
            self._neo4j_client = Neo4jKnowledgeGraphClient(
                uri=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password
            )
            # Setup schema on first run
            if self._neo4j_client.driver:
                self._neo4j_client.setup_schema()
        return self._neo4j_client


    # ---------- Image Extraction ----------
    def extract_images(self):
        manifest = {"pages": []}

        for i, page in enumerate(self.doc):
            img_list = page.get_images(full=True)
            imgs, img_counter = [], 1

            for img in img_list:
                xref = img[0]
                base = self.doc.extract_image(xref)

                if not self.is_proper_image(base):
                    continue

                ext = base.get("ext", "png")
                img_file = self.img_dir / f"page-{i+1:03d}-img-{img_counter:03d}.{ext}"
                img_file.write_bytes(base["image"])

                imgs.append({
                    "index": img_counter,
                    "file": str(img_file),
                    "width": base["width"],
                    "height": base["height"],
                    "ext": ext,
                    "xref": xref,
                    "size_bytes": len(base["image"]),
                    "aspect_ratio": round(base["width"] / base["height"], 2) if base["height"] else 0
                })
                img_counter += 1

            manifest["pages"].append({"page": i+1, "images": imgs})
        return manifest

    @staticmethod
    def is_proper_image(base):
        """Filter out small, decorative, or trivial images."""
        width, height = base.get("width", 0), base.get("height", 0)
        size_bytes = len(base.get("image", b""))

        if width < 50 or height < 50:
            return False
        if size_bytes < 1000:  # <1KB
            return False
        aspect_ratio = width / height if height else 0
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            return False
        if width * height < 2500:  # <50x50 pixels
            return False
        return True

    # ---------- Unified Markdown Document Creation ----------
    def create_unified_markdown_document(self):
        """Create a unified markdown document combining text and tables from the PDF."""
        manifest = {"pages": [], "unified_document": None, "chunks": []}
        
        # Get the full markdown from pymupdf4llm (includes both text and tables)
        full_markdown = pymupdf4llm.to_markdown(str(self.pdf_path))
        
        # Split by pages to maintain page structure
        pages = full_markdown.split('\n---\n')
        all_chunks = []
        
        for i, page_md in enumerate(pages, start=1):
            # Clean up the page markdown
            page_md = page_md.strip()
            if not page_md:
                continue
                
            # Save individual page markdown
            page_file = self.text_dir / f"page-{i:03d}.md"
            page_file.write_text(page_md, encoding="utf-8")
            
            # Count tables in this page
            table_count = len(self.parse_markdown_tables(page_md))
            
            entry = {
                "page": i,
                "file": str(page_file),
                "table_count": table_count,
                "content_length": len(page_md)
            }
            
            # Create chunks for this page if chunking is enabled
            if self.enable_chunking and page_md:
                page_metadata = {
                    'page': i,
                    'source_file': str(self.pdf_path),
                    'content_type': 'markdown'
                }
                page_chunks = self.text_chunker.chunk_markdown(page_md, page_metadata)
                all_chunks.extend(page_chunks)
                entry["chunks"] = len(page_chunks)
            
            manifest["pages"].append(entry)
        
        # Save the unified markdown document
        unified_file = self.text_dir / "unified_document.md"
        unified_file.write_text(full_markdown, encoding="utf-8")
        manifest["unified_document"] = str(unified_file)
        
        # Save all chunks if chunking is enabled
        if self.enable_chunking and all_chunks:
            chunks_file = self.chunks_dir / "unified_chunks.json"
            self.save_json(chunks_file, all_chunks)
            manifest["chunks"] = {
                "unified_chunks_file": str(chunks_file),
                "total_chunks": len(all_chunks),
                "chunks": all_chunks
            }
        
        return manifest


    @staticmethod
    def parse_markdown_tables(md_text: str):
        """Extract markdown tables into list of lists."""
        tables, lines, i = [], md_text.splitlines(), 0
        while i < len(lines):
            if "|" in lines[i] and lines[i].count("|") >= 2:
                table_lines = []
                while i < len(lines) and "|" in lines[i].strip():
                    line = lines[i].strip()
                    if line:
                        table_lines.append(line)
                    i += 1

                if len(table_lines) >= 2:
                    table_data = []
                    for tl in table_lines:
                        if re.match(r'^[\|\-\s:]+$', tl):
                            continue
                        cells = [c.strip() for c in tl.split("|") if c.strip()]
                        if cells:
                            table_data.append(cells)
                    if len(table_data) >= 2:
                        tables.append(table_data)
            else:
                i += 1
        return tables


    # ---------- Indicator Extraction ----------
    def extract_indicators_from_text(self, text: str) -> Dict[str, Any]:
        """Extract indicators from text content."""
        if not self.extract_indicators:
            return {}
        
        return self.indicator_extractor.extract_indicators(text)
    
    def extract_indicators_from_pdf(self) -> Dict[str, Any]:
        """Extract indicators from all text in the PDF."""
        if not self.extract_indicators:
            return {}
        
        manifest = {"pages": [], "summary": {}}
        all_indicators = {
            'domains': [],
            'urls': [],
            'ip_addresses': [],
            'phone_numbers': [],
            'emails': [],
            'social_media': [],
            'tracking_ids': []
        }
        
        # Extract indicators from each page
        for i, page in enumerate(self.doc):
            page_text = page.get_text("text")
            page_indicators = self.extract_indicators_from_text(page_text)
            
            # Save page-level indicators
            page_file = self.indicators_dir / f"page-{i+1:03d}-indicators.json"
            self.save_json(page_file, page_indicators)
            
            manifest["pages"].append({
                "page": i+1,
                "file": str(page_file),
                "counts": {k: len(v) for k, v in page_indicators.items()}
            })
            
            # Aggregate all indicators
            for indicator_type, indicators in page_indicators.items():
                all_indicators[indicator_type].extend(indicators)
        
        # Remove duplicates from aggregated indicators
        for indicator_type in all_indicators:
            seen = set()
            unique_indicators = []
            for indicator in all_indicators[indicator_type]:
                key = indicator.get('normalized', indicator.get('value', ''))
                if key not in seen:
                    seen.add(key)
                    unique_indicators.append(indicator)
            all_indicators[indicator_type] = unique_indicators
        
        # Save aggregated indicators
        all_indicators_file = self.indicators_dir / "all_indicators.json"
        self.save_json(all_indicators_file, all_indicators)
        
        # Create summary
        manifest["summary"] = {
            "total_counts": {k: len(v) for k, v in all_indicators.items()},
            "all_indicators_file": str(all_indicators_file)
        }
        
        # Save manifest
        manifest_file = self.indicators_dir / "indicators_manifest.json"
        self.save_json(manifest_file, manifest)
        
        return manifest

    # ---------- Threat Actor Extraction ----------
    def extract_threat_actors_from_pdf(self) -> Dict[str, Any]:
        """Extract threat actors from all text in the PDF using spaCy NER."""
        if not self.extract_threat_actors:
            return {}
        
        manifest = {"pages": [], "summary": {}}
        all_threat_actors = []
        all_organizations = []
        all_persons = []
        all_locations = []
        all_attack_indicators = []
        
        # Extract threat actors from each page
        for i, page in enumerate(self.doc):
            page_text = page.get_text("text")
            page_result = self.threat_actor_extractor.extract_threat_actors(page_text, i+1)
            
            # Save page-level threat actors
            page_file = self.threat_actors_dir / f"page-{i+1:03d}-threat-actors.json"
            self.save_json(page_file, page_result)
            
            manifest["pages"].append({
                "page": i+1,
                "file": str(page_file),
                "counts": {
                    "threat_actors": page_result["summary"]["total_threat_actors"],
                    "organizations": page_result["summary"]["total_organizations"],
                    "persons": page_result["summary"]["total_persons"],
                    "locations": page_result["summary"]["total_locations"],
                    "attack_indicators": page_result["summary"]["total_attack_indicators"]
                }
            })
            
            # Aggregate all threat actors
            all_threat_actors.extend(page_result["threat_actors"])
            all_organizations.extend(page_result["organizations"])
            all_persons.extend(page_result["persons"])
            all_locations.extend(page_result["locations"])
            all_attack_indicators.extend(page_result["attack_indicators"])
        
        # Remove duplicates from aggregated threat actors
        all_threat_actors = self.threat_actor_extractor._deduplicate_entities(all_threat_actors)
        all_organizations = self.threat_actor_extractor._deduplicate_entities(all_organizations)
        all_persons = self.threat_actor_extractor._deduplicate_entities(all_persons)
        all_locations = self.threat_actor_extractor._deduplicate_entities(all_locations)
        all_attack_indicators = self.threat_actor_extractor._deduplicate_entities(all_attack_indicators)
        
        # Save aggregated threat actors
        all_threat_actors_data = {
            'threat_actors': all_threat_actors,
            'organizations': all_organizations,
            'persons': all_persons,
            'locations': all_locations,
            'attack_indicators': all_attack_indicators
        }
        all_threat_actors_file = self.threat_actors_dir / "all_threat_actors.json"
        self.save_json(all_threat_actors_file, all_threat_actors_data)
        
        # Create summary
        manifest["summary"] = {
            "total_counts": {
                "threat_actors": len(all_threat_actors),
                "organizations": len(all_organizations),
                "persons": len(all_persons),
                "locations": len(all_locations),
                "attack_indicators": len(all_attack_indicators)
            },
            "all_threat_actors_file": str(all_threat_actors_file)
        }
        
        # Save manifest
        manifest_file = self.threat_actors_dir / "threat_actors_manifest.json"
        self.save_json(manifest_file, manifest)
        
        return manifest

    # ---------- Neo4j Knowledge Graph Ingestion ----------
    def ingest_to_neo4j(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest document data into Neo4j knowledge graph."""
        if not self.enable_neo4j or not self.neo4j_client.driver:
            return {"error": "Neo4j not enabled or not available"}
        
        results = {
            "document_created": False,
            "indicators_created": 0,
            "chunks_created": 0,
            "relationships_created": 0,
            "errors": []
        }
        
        try:
            # Create document node
            filename = self.pdf_path.name
            file_hash = hashlib.md5(self.pdf_path.stem.encode()).hexdigest()
            
            document_data = {
                "id": str(uuid.uuid4()),
                "filename": filename,
                "file_path": str(self.pdf_path),
                "file_hash": file_hash,
                "file_size": self.pdf_path.stat().st_size if self.pdf_path.exists() else 0,
                "page_count": len(self.doc),
                "language": "en",
                "document_type": "pdf",
                "processing_status": "completed",
                "extracted_text_length": 0,  # Will be calculated
                "chunk_count": 0,  # Will be calculated
                "table_count": len(manifest.get("tables", {}).get("pages", [])),
                "image_count": len(manifest.get("images", {}).get("pages", [])),
                "confidence_score": 0.9,
                "source": filename
            }
            
            document_id = self.neo4j_client.create_document_node(document_data)
            if document_id:
                results["document_created"] = True
                print(f"Created document node: {document_id}")
            else:
                results["errors"].append("Failed to create document node")
                return results
            
            # Process indicators
            indicators_data = manifest.get("indicators", {})
            if indicators_data and "summary" in indicators_data:
                all_indicators_file = indicators_data.get("summary", {}).get("all_indicators_file")
                if all_indicators_file and Path(all_indicators_file).exists():
                    try:
                        with open(all_indicators_file, 'r') as f:
                            indicators = json.load(f)
                        
                        created_indicators = []
                        for indicator_type, indicator_list in indicators.items():
                            for indicator_data in indicator_list:
                                try:
                                    # Map indicator type to Neo4j format
                                    type_mapping = {
                                        'domains': 'domain',
                                        'urls': 'url',
                                        'ip_addresses': 'ip_address',
                                        'emails': 'email',
                                        'phone_numbers': 'phone',
                                        'social_media': 'social_media',
                                        'tracking_ids': 'tracking_id'
                                    }
                                    
                                    neo4j_type = type_mapping.get(indicator_type, 'domain')
                                    
                                    indicator_node_data = {
                                        "id": str(uuid.uuid4()),
                                        "value": indicator_data.get("value", ""),
                                        "normalized_value": indicator_data.get("normalized", ""),
                                        "indicator_type": neo4j_type,
                                        "context": indicator_data.get("context", ""),
                                        "page_number": indicator_data.get("page_number"),
                                        "chunk_id": indicator_data.get("chunk_id"),
                                        "extraction_method": "regex",
                                        "threat_level": "low",
                                        "is_verified": False,
                                        "verification_date": None,
                                        "verification_source": "",
                                        "confidence_score": 0.8,
                                        "source": filename
                                    }
                                    
                                    indicator_id = self.neo4j_client.create_indicator_node(indicator_node_data)
                                    if indicator_id:
                                        created_indicators.append(indicator_id)
                                        
                                        # Create document-indicator relationship
                                        self.neo4j_client.create_relationship(
                                            document_id, indicator_id, "CONTAINS",
                                            {"extraction_context": indicator_data.get("context", ""),
                                             "extraction_confidence": 0.8}
                                        )
                                        results["relationships_created"] += 1
                                    
                                except Exception as e:
                                    results["errors"].append(f"Failed to create indicator: {e}")
                        
                        results["indicators_created"] = len(created_indicators)
                        print(f"Created {len(created_indicators)} indicator nodes")
                    
                    except Exception as e:
                        results["errors"].append(f"Failed to load indicators: {e}")
            
            # Process chunks if available
            unified_collection = manifest.get("unified_collection", {})
            if unified_collection and "chunks" in unified_collection:
                chunks_data = unified_collection["chunks"]
                created_chunks = []
                
                for chunk_data in chunks_data:
                    try:
                        chunk_node_data = {
                            "id": str(uuid.uuid4()),
                            "content": chunk_data.get("content", ""),
                            "chunk_index": chunk_data.get("chunk_index", 0),
                            "content_type": chunk_data.get("content_type", "text"),
                            "page_number": chunk_data.get("page", 0),
                            "chunk_size": chunk_data.get("chunk_size", 0),
                            "embedding_id": chunk_data.get("id"),
                            "confidence_score": 0.9,
                            "source": filename
                        }
                        
                        chunk_id = self.neo4j_client.create_chunk_node(chunk_node_data)
                        if chunk_id:
                            created_chunks.append(chunk_id)
                            
                            # Create document-chunk relationship
                            self.neo4j_client.create_relationship(
                                document_id, chunk_id, "CONTAINS"
                            )
                            results["relationships_created"] += 1
                    
                    except Exception as e:
                        results["errors"].append(f"Failed to create chunk: {e}")
                
                results["chunks_created"] = len(created_chunks)
                print(f"Created {len(created_chunks)} chunk nodes")
        
        except Exception as e:
            results["errors"].append(f"Failed to ingest to Neo4j: {e}")
            print(f"Neo4j ingestion failed: {e}")
        
        return results

    # ---------- Main Process ----------
    def process(self):
        manifest = {
            "pdf": str(self.pdf_path),
            "out": str(self.pdf_dir)
        }
        
        unified_markdown_manifest = None
        
        try:
            # Extract unified markdown document (combines text and tables)
            unified_markdown_manifest = self.create_unified_markdown_document()
            manifest["unified_markdown"] = unified_markdown_manifest
        except Exception as e:
            manifest["unified_markdown_error"] = str(e)

        try:
            manifest["images"] = self.extract_images()
        except Exception as e:
            manifest["images_error"] = str(e)

        try:
            if self.extract_indicators:
                manifest["indicators"] = self.extract_indicators_from_pdf()
        except Exception as e:
            manifest["indicators_error"] = str(e)

        try:
            if self.extract_threat_actors:
                manifest["threat_actors"] = self.extract_threat_actors_from_pdf()
        except Exception as e:
            manifest["threat_actors_error"] = str(e)

        # Use unified markdown chunks as the unified collection
        unified_collection = None
        try:
            if self.enable_chunking and unified_markdown_manifest and "chunks" in unified_markdown_manifest:
                # The unified markdown already contains chunks, so we use them directly
                unified_collection = {
                    'unified_collection_file': unified_markdown_manifest["chunks"]["unified_chunks_file"],
                    'metadata': {
                        'total_chunks': unified_markdown_manifest["chunks"]["total_chunks"],
                        'text_chunks': unified_markdown_manifest["chunks"]["total_chunks"],  # All chunks are now markdown
                        'table_chunks': 0,  # Tables are now part of markdown chunks
                        'pages_covered': sorted(list(set(c.get('page', 0) for c in unified_markdown_manifest["chunks"]["chunks"]))),
                        'created_at': datetime.now().isoformat(),
                        'source_file': str(self.pdf_path),
                        'chunk_size': self.text_chunker.chunk_size,
                        'chunk_overlap': self.text_chunker.chunk_overlap
                    },
                    'chunks': unified_markdown_manifest["chunks"]["chunks"]
                }
                manifest["unified_collection"] = unified_collection
        except Exception as e:
            manifest["unified_collection_error"] = str(e)

        # Upsert to Qdrant if enabled and we have chunks
        try:
            if self.enable_qdrant and unified_collection and "chunks" in unified_collection:
                qdrant_result = self.qdrant_store.upsert_chunks(unified_collection["chunks"])
                manifest["qdrant_upsert"] = qdrant_result
        except Exception as e:
            manifest["qdrant_upsert_error"] = str(e)

        # Upsert images to Qdrant if enabled
        try:
            if self.enable_qdrant and "images" in manifest and "pages" in manifest["images"]:
                # Collect all images from all pages
                all_images = []
                for page_data in manifest["images"]["pages"]:
                    if "images" in page_data:
                        for image_data in page_data["images"]:
                            # Add source file information to image data
                            image_data["source_file"] = str(self.pdf_path)
                            all_images.append(image_data)
                
                if all_images:
                    qdrant_image_result = self.qdrant_store.upsert_images(all_images)
                    manifest["qdrant_image_upsert"] = qdrant_image_result
        except Exception as e:
            manifest["qdrant_image_upsert_error"] = str(e)

        # Extract Neo4j knowledge graph data
        try:
            neo4j_data = self.neo4j_data_extractor.extract_neo4j_data(manifest)
            neo4j_data_file = self.neo4j_data_extractor.save_neo4j_data(neo4j_data, self.neo4j_dir)
            manifest["neo4j_data"] = {
                "neo4j_data_file": neo4j_data_file,
                "total_nodes": neo4j_data["metadata"]["total_nodes"],
                "total_edges": neo4j_data["metadata"]["total_edges"],
                "node_breakdown": {
                    "indicators": len(neo4j_data["nodes"]["indicators"]),
                    "documents": len(neo4j_data["nodes"]["documents"]),
                    "campaigns": len(neo4j_data["nodes"]["campaigns"]),
                    "threat_actors": len(neo4j_data["nodes"]["threat_actors"])
                },
                "edge_breakdown": {
                    "mentioned_in": len(neo4j_data["edges"]["mentioned_in"]),
                    "related_to": len(neo4j_data["edges"]["related_to"]),
                    "part_of_campaign": len(neo4j_data["edges"]["part_of_campaign"])
                }
            }
        except Exception as e:
            manifest["neo4j_data_error"] = str(e)

        # Ingest to Neo4j knowledge graph if enabled
        # NOTE: Disabled in favor of dedicated Neo4j upserter in API
        # try:
        #     if self.enable_neo4j and self.neo4j_client.driver:
        #         neo4j_result = self.ingest_to_neo4j(manifest)
        #         manifest["neo4j_ingestion"] = neo4j_result
        # except Exception as e:
        #     manifest["neo4j_ingestion_error"] = str(e)

        self.save_json(self.pdf_dir / "manifest.json", manifest)
        return manifest
