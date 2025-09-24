#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified extraction from PDFs with chunking and vector database support:
- Text (reading-order + optional blocks) [PyMuPDF]
- Embedded images [PyMuPDF]
- Tables [pymupdf4llm markdown + pandas]
- Text and table chunking with configurable parameters
- Unified collection combining text and table chunks
- Indicator extraction (domains, URLs, emails, etc.)
- Qdrant vector database integration with embeddings
- Neo4j knowledge graph integration with relationship discovery
- Semantic search capabilities
"""

import argparse
import json
import glob
import traceback
from pathlib import Path

# Import from shared modules
import sys
sys.path.append('/Users/akarna/Blackbox POC/src')

from shared.clients.neo4j_client import Neo4jKnowledgeGraphClient
from shared.services.parse_pdf import PDFExtractor


# ---------------- CLI ----------------
def collect_pdfs(input_path: Path, recursive: bool):
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    if input_path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        return list(input_path.glob(pattern))
    return [Path(p) for p in glob.glob(str(input_path)) if p.lower().endswith(".pdf")]


def main():
    ap = argparse.ArgumentParser(description="Extract text + images + tables + indicators from PDFs with chunking, Qdrant, and Neo4j support.")
    ap.add_argument("--input", required=True, help="PDF file, folder, or glob")
    ap.add_argument("--out", default="output", help="Output folder")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders if input is a dir")
    ap.add_argument("--blocks", action="store_true", help="Also dump layout-aware block JSON")
    ap.add_argument("--no-indicators", action="store_true", help="Skip indicator extraction")
    ap.add_argument("--no-threat-actors", action="store_true", help="Skip threat actor extraction using NER")
    ap.add_argument("--no-chunking", action="store_true", help="Disable text and table chunking")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters (default: 1000)")
    ap.add_argument("--chunk-overlap", type=int, default=250, help="Chunk overlap in characters (default: 250, 25% of chunk size)")
    ap.add_argument("--enable-qdrant", action="store_true", help="Enable Qdrant vector database upsert")
    ap.add_argument("--qdrant-host", default="localhost", help="Qdrant host (default: localhost)")
    ap.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port (default: 6333)")
    ap.add_argument("--qdrant-collection", default="document_chunks", help="Qdrant collection name (default: document_chunks)")
    ap.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Embedding model (default: multilingual)")
    ap.add_argument("--qdrant-api-key", help="Qdrant API key (optional)")
    ap.add_argument("--enable-neo4j", action="store_true", help="Enable Neo4j knowledge graph ingestion")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j connection URI (default: bolt://localhost:7687)")
    ap.add_argument("--neo4j-username", default="neo4j", help="Neo4j username (default: neo4j)")
    ap.add_argument("--neo4j-password", default="testpass", help="Neo4j password (default: testpass)")
    ap.add_argument("--discover-relationships", action="store_true", help="Discover co-occurrence relationships after processing")
    ap.add_argument("--min-co-occurrences", type=int, default=2, help="Minimum co-occurrences for relationship discovery (default: 2)")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    pdfs = collect_pdfs(Path(args.input), args.recursive)
    if not pdfs:
        raise SystemExit("No PDFs found")

    summary = []
    for pdf in pdfs:
        print(f"[+] Processing {pdf}")
        try:
            extractor = PDFExtractor(
                pdf, 
                out_root, 
                args.blocks, 
                not args.no_indicators,
                not args.no_chunking,
                args.chunk_size,
                args.chunk_overlap,
                args.enable_qdrant,
                args.qdrant_host,
                args.qdrant_port,
                args.qdrant_collection,
                args.embedding_model,
                args.qdrant_api_key,
                args.enable_neo4j,
                args.neo4j_uri,
                args.neo4j_username,
                args.neo4j_password,
                not args.no_threat_actors
            )
            result = extractor.process()
            summary.append(result)
            
            # Print indicator summary if extracted
            if not args.no_indicators and "indicators" in result:
                indicator_summary = result["indicators"].get("summary", {})
                total_counts = indicator_summary.get("total_counts", {})
                if any(total_counts.values()):
                    print(f"    Found indicators: {total_counts}")
            
            # Print threat actor summary if extracted
            if not args.no_threat_actors and "threat_actors" in result:
                threat_actor_summary = result["threat_actors"].get("summary", {})
                total_counts = threat_actor_summary.get("total_counts", {})
                if any(total_counts.values()):
                    print(f"    Found threat actors: {total_counts}")
            
            # Print Neo4j knowledge graph data summary
            if "neo4j_data" in result:
                neo4j_data = result["neo4j_data"]
                print(f"    Neo4j knowledge graph: {neo4j_data.get('total_nodes', 0)} nodes, "
                      f"{neo4j_data.get('total_edges', 0)} edges")
                node_breakdown = neo4j_data.get('node_breakdown', {})
                edge_breakdown = neo4j_data.get('edge_breakdown', {})
                if any(node_breakdown.values()):
                    print(f"      Nodes: {node_breakdown}")
                if any(edge_breakdown.values()):
                    print(f"      Edges: {edge_breakdown}")
            
            # Print chunking summary if chunking is enabled
            if not args.no_chunking and "unified_collection" in result:
                collection_metadata = result["unified_collection"].get("metadata", {})
                print(f"    Created {collection_metadata.get('total_chunks', 0)} chunks "
                      f"({collection_metadata.get('text_chunks', 0)} text, "
                      f"{collection_metadata.get('table_chunks', 0)} table)")
            
            # Print Qdrant upsert summary if enabled
            if args.enable_qdrant and "qdrant_upsert" in result:
                qdrant_result = result["qdrant_upsert"]
                print(f"    Qdrant chunks: {qdrant_result.get('processed', 0)} chunks processed, "
                      f"{qdrant_result.get('errors', 0)} errors")
                if qdrant_result.get('errors', 0) > 0:
                    print(f"    Qdrant chunk errors: {qdrant_result.get('error_details', [])[:3]}")  # Show first 3 errors
            
            # Print Qdrant image upsert summary if enabled
            if args.enable_qdrant and "qdrant_image_upsert" in result:
                qdrant_image_result = result["qdrant_image_upsert"]
                print(f"    Qdrant images: {qdrant_image_result.get('processed', 0)} images processed, "
                      f"{qdrant_image_result.get('errors', 0)} errors")
                if qdrant_image_result.get('errors', 0) > 0:
                    print(f"    Qdrant image errors: {qdrant_image_result.get('error_details', [])[:3]}")  # Show first 3 errors
            
            # Print Neo4j ingestion summary if enabled
            if args.enable_neo4j and "neo4j_ingestion" in result:
                neo4j_result = result["neo4j_ingestion"]
                print(f"    Neo4j ingestion: {neo4j_result.get('document_created', False)} document, "
                      f"{neo4j_result.get('indicators_created', 0)} indicators, "
                      f"{neo4j_result.get('chunks_created', 0)} chunks, "
                      f"{neo4j_result.get('relationships_created', 0)} relationships")
                if neo4j_result.get('errors'):
                    print(f"    Neo4j errors: {neo4j_result.get('errors', [])[:3]}")  # Show first 3 errors
        except Exception as e:
            print(f"[!] Error on {pdf}: {e}")
            print(f"Traceback: {traceback.format_exc()}")

    # Discover relationships if enabled
    if args.discover_relationships and args.enable_neo4j:
        print("\n[+] Discovering co-occurrence relationships...")
        try:
            # Create a temporary client for relationship discovery
            temp_client = Neo4jKnowledgeGraphClient(
                uri=args.neo4j_uri,
                username=args.neo4j_username,
                password=args.neo4j_password
            )
            if temp_client.driver:
                rel_result = temp_client.discover_co_occurrence_relationships(args.min_co_occurrences)
                print(f"    Discovered {rel_result.get('relationships_created', 0)} co-occurrence relationships")
            temp_client.close()
        except Exception as e:
            print(f"[!] Error discovering relationships: {e}")

    # Save run summary
    summary_file = out_root / "run_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Done. Results in {out_root}")


if __name__ == "__main__":
    main()
