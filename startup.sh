#!/bin/bash

# Blackbox POC - PDF Processing Startup Script
# This script processes all PDFs in the src/docs directory with full pipeline

set -e  # Exit on any error

echo "🚀 Starting Blackbox POC PDF Processing Pipeline"
echo "=================================================="

# Pre-load embedding models for fast startup
echo "📥 Pre-loading embedding models..."
python preload_models.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to pre-load models. Continuing anyway..."
fi
echo ""

# Configuration
PDF_DIR="src/docs"
OUTPUT_DIR="output"
PYTHON_ENV="blackbox-poc-notebook-2"

# Check if pyenv is available
if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv not found. Please install pyenv first."
    exit 1
fi

# Activate Python environment
echo "🐍 Activating Python environment: $PYTHON_ENV"
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate $PYTHON_ENV

# Check if the environment exists
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate Python environment: $PYTHON_ENV"
    echo "Please create the environment first with: pyenv virtualenv 3.13.5 $PYTHON_ENV"
    exit 1
fi

# Check if PDF directory exists
if [ ! -d "$PDF_DIR" ]; then
    echo "❌ PDF directory not found: $PDF_DIR"
    exit 1
fi

# Find all PDF files
PDF_FILES=($(find "$PDF_DIR" -name "*.pdf" -type f))
if [ ${#PDF_FILES[@]} -eq 0 ]; then
    echo "❌ No PDF files found in $PDF_DIR"
    exit 1
fi

echo "📄 Found ${#PDF_FILES[@]} PDF file(s) to process:"
for pdf in "${PDF_FILES[@]}"; do
    echo "  - $(basename "$pdf")"
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each PDF
echo ""
echo "🔄 Starting PDF processing..."
echo "=============================="

TOTAL_PROCESSED=0
TOTAL_FAILED=0

for pdf_file in "${PDF_FILES[@]}"; do
    pdf_name=$(basename "$pdf_file")
    echo ""
    echo "📋 Processing: $pdf_name"
    echo "----------------------------------------"
    
    # Run preprocessing with full pipeline
    if python notebooks/preprocess.py \
        --input "$pdf_file" \
        --out "$OUTPUT_DIR" \
        --enable-neo4j \
        --enable-qdrant \
        --qdrant-api-key "nYvmqn8eYkq8cHeLGk5Vj_px3AzXGRkIkEbxt6virSJ-8uih0JJUQw" \
        --chunk-size 1000 \
        --chunk-overlap 250; then
        
        echo "✅ Successfully processed: $pdf_name"
        TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))
    else
        echo "❌ Failed to process: $pdf_name"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
done

# Summary
echo ""
echo "📊 Processing Summary"
echo "===================="
echo "✅ Successfully processed: $TOTAL_PROCESSED PDF(s)"
echo "❌ Failed to process: $TOTAL_FAILED PDF(s)"
echo "📁 Output directory: $OUTPUT_DIR"

# Show output structure
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "📂 Output Structure:"
    echo "==================="
    for pdf_dir in "$OUTPUT_DIR"/pdf_*; do
        if [ -d "$pdf_dir" ]; then
            pdf_name=$(basename "$pdf_dir")
            echo "  📁 $pdf_name/"
            
            # Count files in each subdirectory
            for subdir in "$pdf_dir"/*; do
                if [ -d "$subdir" ]; then
                    subdir_name=$(basename "$subdir")
                    file_count=$(find "$subdir" -type f | wc -l)
                    echo "    📁 $subdir_name/ ($file_count files)"
                fi
            done
        fi
    done
fi

# Upsert Neo4j data for all processed PDFs
echo ""
echo "🔄 Upserting Neo4j knowledge graphs..."
echo "======================================"
for pdf_dir in "$OUTPUT_DIR"/pdf_*; do
    if [ -d "$pdf_dir" ]; then
        pdf_name=$(basename "$pdf_dir")
        neo4j_file="$pdf_dir/neo4j/neo4j_knowledge_graph.json"
        if [ -f "$neo4j_file" ]; then
            echo "📋 Upserting Neo4j data for: $pdf_name"
            python notebooks/upsert_to_neo4j.py --file "$neo4j_file" > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo "  ✅ Successfully upserted: $pdf_name"
            else
                echo "  ❌ Failed to upsert: $pdf_name"
            fi
        fi
    fi
done

# Neo4j Database Summary
echo ""
echo "🗄️  Neo4j Database Summary"
echo "=========================="
if command -v python &> /dev/null; then
    python -c "
from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'testpass'))
    with driver.session() as session:
        # Check total relationships
        result = session.run('MATCH ()-[r]->() RETURN count(r) as total_rels')
        total_rels = result.single()['total_rels']
        print(f'📊 Total relationships: {total_rels}')
        
        # Check relationship types
        result = session.run('MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC')
        print('🔗 Relationship types:')
        for record in result:
            print(f'  - {record[\"rel_type\"]}: {record[\"count\"]}')
        
        # Check nodes
        result = session.run('MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC')
        print('📋 Node counts:')
        for record in result:
            print(f'  - {record[\"label\"]}: {record[\"count\"]}')
        
        # Check campaigns
        result = session.run('MATCH (c:Campaign) RETURN c.name as campaign_name ORDER BY c.name')
        print('🎯 Campaigns:')
        for record in result:
            print(f'  - {record[\"campaign_name\"]}')
    
    driver.close()
    print('✅ Neo4j connection successful')
except Exception as e:
    print(f'❌ Neo4j connection failed: {e}')
    print('💡 Make sure Neo4j is running: docker-compose up neo4j')
"
else
    echo "❌ Python not available for Neo4j check"
fi

echo ""
echo "🎉 PDF Processing Pipeline Complete!"
echo "===================================="

if [ $TOTAL_FAILED -eq 0 ]; then
    echo "✅ All PDFs processed successfully!"
    echo ""
    echo "🔍 Next steps:"
    echo "  - Check the output directory: $OUTPUT_DIR"
    echo "  - View Neo4j database at: http://localhost:7474"
    echo "  - Access Qdrant at: http://localhost:6333"
    echo "  - Run API server: python src/main.py"
else
    echo "⚠️  Some PDFs failed to process. Check the logs above for details."
    exit 1
fi
