# Blackbox POC - Document Intelligence Pipeline

A sophisticated document intelligence pipeline that combines multiple AI technologies to extract, analyze, and search through threat intelligence documents. This system uses LangGraph for intelligent query routing, Neo4j for knowledge graph storage, and Qdrant for vector similarity search.

## 🚀 Features

### Core Capabilities
- **Intelligent Query Routing**: Uses LangGraph to automatically route queries to the most appropriate search method
- **Multi-Modal Search**: Combines vector search, graph traversal, pattern detection, and semantic analysis
- **Threat Intelligence Extraction**: Automatically extracts indicators, threat actors, and campaign information from documents
- **Network Analysis**: Performs hop-based network analysis to find connected entities
- **Timeline Analysis**: Shows indicator relationships and patterns over time
- **Cross-Campaign Analysis**: Identifies indicators that appear across multiple campaigns

### Search Types Supported
1. **Vector Search**: Semantic similarity search using embeddings
2. **Graph Search**: Neo4j knowledge graph traversal and network analysis
3. **Indicator Search**: Targeted lookup of specific indicators (domains, URLs, emails, etc.)
4. **Pattern Detection**: Clustering and pattern analysis
5. **Campaign Analysis**: Cross-document campaign intelligence
6. **Timeline Analysis**: Temporal relationship analysis

## 🏗️ Architecture

### Technology Stack
- **Backend**: FastAPI with Python 3.11
- **AI Framework**: LangGraph for intelligent routing
- **Vector Database**: Qdrant for similarity search
- **Graph Database**: Neo4j with APOC and Graph Data Science plugins
- **Document Processing**: PyMuPDF4LLM for PDF parsing
- **Embeddings**: Sentence Transformers
- **Containerization**: Docker Compose

### System Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Qdrant        │    │   Neo4j         │
│   (Port 5006)   │◄──►│   (Port 6333)   │    │   (Port 7474)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   LangGraph     │
│   RAG Agent     │
└─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- Poetry (for dependency management)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Blackbox POC"
   ```

2. **Start the services**
   ```bash
   docker compose up -d
   ```

3. **Wait for services to initialize** (approximately 2-3 minutes)

4. **Verify the installation**
   ```bash
   curl http://localhost:5006/search?query=test
   ```

### Access Points
- **API**: http://localhost:5006
- **Neo4j Browser**: http://localhost:7474 (username: neo4j, password: testpass)
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📖 API Usage

### Main Search Endpoint

**GET** `/search`

Intelligent search with automatic query classification and routing.

#### Parameters
- `query` (required): Search query string
- `limit` (optional): Maximum results (1-100, default: 10)
- `score_threshold` (optional): Score threshold (0.0-1.0, default: 0.7)

#### Example Queries

**Vector Search (Semantic)**
```bash
curl "http://localhost:5006/search?query=Russian disinformation campaigns"
```

**Indicator Search**
```bash
curl "http://localhost:5006/search?query=Find all domains associated with Doppelgänger"
```

**Network Analysis**
```bash
curl "http://localhost:5006/search?query=Show all indicators within 2 hops of domain disinfo"
```

**Timeline Analysis**
```bash
curl "http://localhost:5006/search?query=Show indicator relationships over time"
```

**Pattern Detection**
```bash
curl "http://localhost:5006/search?query=Find clusters of related social media accounts"
```

**Campaign Analysis**
```bash
curl "http://localhost:5006/search?query=Which indicators appear across multiple campaigns?"
```

#### Response Format
```json
{
  "query": "Find all domains associated with Doppelgänger",
  "search_type": "intelligent_routing",
  "query_classification": {
    "query_type": "indicator_search",
    "query_intent": "indicator_lookup",
    "search_strategy": "targeted"
  },
  "response": "Found results for 'Find all domains associated with Doppelgänger'",
  "confidence_score": 1.0,
  "results": {
    "vector_results": [],
    "graph_results": [],
    "indicator_results": [...],
    "pattern_results": [],
    "campaign_results": [],
    "timeline_results": []
  },
  "metadata": {
    "total_vector_results": 0,
    "total_graph_results": 0,
    "total_indicator_results": 50,
    "total_pattern_results": 0,
    "total_campaign_results": 0,
    "total_timeline_results": 0
  },
  "parameters": {
    "limit": 10,
    "score_threshold": 0.7
  },
  "status": "success"
}
```

### Additional Endpoints

**GET** `/indicator-type`
- Get the type of a specific indicator

**GET** `/full-context`
- Get full context and relationships for an indicator

**GET** `/graph-traversal`
- Perform graph traversal with connection analysis

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpass
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_API_KEY=your_api_key_here
```

### Docker Services Configuration

The system uses three main services:

1. **App Service** (Port 5006)
   - FastAPI application
   - LangGraph RAG agent
   - Document processing pipeline

2. **Qdrant Service** (Port 6333)
   - Vector database for embeddings
   - Similarity search capabilities

3. **Neo4j Service** (Port 7474)
   - Graph database for knowledge graph
   - APOC and Graph Data Science plugins

## 📊 Data Processing Pipeline

### Document Processing Flow
1. **PDF Upload**: Documents are processed using PyMuPDF4LLM
2. **Text Extraction**: Text, images, and tables are extracted
3. **Indicator Extraction**: Domains, URLs, emails, IPs, social media accounts
4. **Threat Actor Extraction**: Named entity recognition for threat actors
5. **Knowledge Graph Construction**: Entities and relationships stored in Neo4j
6. **Vector Embeddings**: Document chunks embedded and stored in Qdrant

### Supported Document Types
- PDF documents
- Text files
- Structured data (JSON)

## 🧠 Intelligent Query Routing

The system uses LangGraph to automatically classify and route queries:

### Query Classification
- **Timeline Queries**: "Show indicator relationships over time"
- **Pattern Detection**: "Find clusters of related accounts"
- **Campaign Analysis**: "Which indicators appear across multiple campaigns?"
- **Indicator Search**: "Find all domains associated with X"
- **Network Analysis**: "Show indicators within N hops of domain X"
- **Vector Search**: General semantic queries

### Search Strategies
- **Temporal**: Timeline and chronological analysis
- **Pattern-focused**: Clustering and pattern detection
- **Cross-document**: Multi-campaign analysis
- **Targeted**: Specific indicator lookup
- **Graph-focused**: Network traversal and relationships
- **Vector-focused**: Semantic similarity search

## 🔍 Advanced Features

### Network Analysis
- **Hop-based Traversal**: Find entities within N hops of a starting point
- **Path Analysis**: Track connection paths between entities
- **Relationship Mapping**: Visualize entity relationships

### Special Character Handling
- **Unicode Normalization**: Handles special characters (ä, ö, ü, etc.)
- **ASCII Conversion**: Converts special characters for better matching
- **Multi-language Support**: Supports various character encodings

### Cross-Campaign Intelligence
- **Indicator Correlation**: Find indicators appearing across multiple campaigns
- **Document Co-occurrence**: Analyze shared indicators between documents
- **Significance Scoring**: Rank indicators by cross-campaign frequency

## 🛠️ Development

### Local Development Setup

1. **Install Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install Dependencies**
   ```bash
   poetry install
   ```

3. **Run Locally**
   ```bash
   poetry run python src/main.py
   ```

### Project Structure
```
Blackbox POC/
├── src/
│   ├── components/rag/          # RAG API components
│   ├── shared/
│   │   ├── clients/             # Database clients
│   │   ├── services/            # Core services
│   │   └── utils/               # Utility functions
│   └── main.py                  # FastAPI application
├── notebooks/                   # Jupyter notebooks for analysis
├── output/                      # Processed document outputs
├── neo4j/                       # Neo4j data and logs
├── qdrant_data/                 # Qdrant vector data
├── docker-compose.yml           # Docker services
├── Dockerfile                   # Application container
└── pyproject.toml              # Python dependencies
```

## 📈 Performance

### Optimization Features
- **Intelligent Caching**: Results cached for repeated queries
- **Parallel Processing**: Concurrent database queries
- **Result Limiting**: Configurable result limits
- **Score Thresholding**: Quality-based result filtering

### Scalability
- **Horizontal Scaling**: Docker-based deployment
- **Database Optimization**: Indexed queries and optimized Cypher
- **Memory Management**: Configurable memory limits for services

## 🔒 Security

### Security Features
- **API Key Authentication**: Qdrant API key protection
- **Database Authentication**: Neo4j user authentication
- **Input Validation**: Query parameter validation
- **Error Handling**: Secure error responses

## 🐛 Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   docker compose down
   docker compose up -d
   ```

2. **Empty search results**
   - Check if documents have been processed
   - Verify Neo4j and Qdrant connections
   - Check service logs: `docker compose logs`

3. **Memory issues**
   - Increase Docker memory limits
   - Check Neo4j memory configuration

### Logs and Debugging
```bash
# View all logs
docker compose logs

# View specific service logs
docker compose logs app
docker compose logs neo4j
docker compose logs qdrant
```

## 📝 License

This project is for proof-of-concept purposes. Please ensure compliance with your organization's policies regarding data processing and AI usage.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review service logs
- Create an issue in the repository

---

**Note**: This is a proof-of-concept system designed for threat intelligence analysis. Ensure proper security measures are in place before deploying in production environments.
