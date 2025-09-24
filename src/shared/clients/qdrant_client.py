from typing import Dict, List, Any
import uuid
import base64
import hashlib
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from shared.services.embedding_model_manager import embedding_manager

class QdrantVectorStore:
    """Handle Qdrant vector database operations for document chunks with lazy initialization."""
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 collection_name: str = "document_chunks", 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 api_key: str = None):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.api_key = api_key
        
        # Lazy initialization
        self._client = None
        self._initialized = False
    
    @property
    def client(self):
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            if self.api_key:
                self._client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key, https=False)
            else:
                self._client = QdrantClient(host=self.host, port=self.port, https=False)
        return self._client
    
    @property
    def embedding_model(self):
        """Get embedding model from manager."""
        return embedding_manager.model
    
    @property
    def embedding_dimension(self):
        """Get embedding dimension from manager."""
        return embedding_manager.embedding_dimension
    
    def _ensure_initialized(self):
        """Ensure client and collection are initialized."""
        if not self._initialized:
            self._ensure_collection_exists()
            self._initialized = True
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            else:
                print(f"Using existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            return embedding_manager.encode_single(text)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string."""
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
                return base64_string
        except Exception as e:
            print(f"Error encoding image {image_path} to base64: {e}")
            raise
    
    def get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract metadata from image file."""
        try:
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Get file stats
            stat = image_file.stat()
            
            # Generate hash for deduplication
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            return {
                'filename': image_file.name,
                'file_path': str(image_path),
                'file_size': stat.st_size,
                'file_hash': file_hash,
                'file_extension': image_file.suffix.lower(),
                'created_at': stat.st_ctime,
                'modified_at': stat.st_mtime
            }
        except Exception as e:
            print(f"Error getting image metadata for {image_path}: {e}")
            raise
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:
        """Upsert chunks to Qdrant vector database."""
        self._ensure_initialized()
        
        if not chunks:
            return {"status": "no_chunks", "processed": 0}
        
        processed_count = 0
        error_count = 0
        errors = []
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            points = []
            
            for chunk in batch:
                try:
                    # Generate embedding
                    embedding = self.generate_embedding(chunk['content'])
                    
                    # Create point for Qdrant
                    point = PointStruct(
                        id=str(uuid.uuid4()),  # Generate new UUID for each upsert
                        vector=embedding,
                        payload={
                            'chunk_id': chunk['id'],
                            'content': chunk['content'],
                            'content_type': chunk['content_type'],
                            'page': chunk['page'],
                            'chunk_index': chunk['chunk_index'],
                            'source_file': chunk['source_file'],
                            'chunk_size': chunk['chunk_size'],
                            'created_at': chunk['created_at'],
                            'metadata': chunk.get('metadata', {})
                        }
                    )
                    points.append(point)
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    errors.append(f"Chunk {chunk.get('id', 'unknown')}: {str(e)}")
                    print(f"Error processing chunk {chunk.get('id', 'unknown')}: {e}")
            
            # Upsert batch to Qdrant
            if points:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    print(f"Upserted batch of {len(points)} chunks to Qdrant")
                except Exception as e:
                    error_count += len(points)
                    errors.append(f"Batch upsert error: {str(e)}")
                    print(f"Error upserting batch: {e}")
        
        return {
            "status": "completed",
            "processed": processed_count,
            "errors": error_count,
            "error_details": errors,
            "collection_name": self.collection_name
        }
    
    def upsert_images(self, images: List[Dict[str, Any]], batch_size: int = 50) -> Dict[str, Any]:
        """Upsert base64 encoded images to Qdrant."""
        self._ensure_initialized()
        if not images:
            return {"status": "no_images", "processed": 0}
        
        processed_count = 0
        error_count = 0
        errors = []
        
        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            points = []
            
            for image_data in batch:
                try:
                    # Get image path
                    image_path = image_data.get('file')
                    if not image_path or not Path(image_path).exists():
                        raise FileNotFoundError(f"Image file not found: {image_path}")
                    
                    # Encode image to base64
                    base64_image = self.encode_image_to_base64(image_path)
                    
                    # Get image metadata
                    image_metadata = self.get_image_metadata(image_path)
                    
                    # Create a text description for embedding (combine available metadata)
                    description_parts = []
                    if image_data.get('page'):
                        description_parts.append(f"Page {image_data['page']}")
                    if image_data.get('index'):
                        description_parts.append(f"Image {image_data['index']}")
                    if image_metadata.get('filename'):
                        description_parts.append(f"File: {image_metadata['filename']}")
                    if image_data.get('width') and image_data.get('height'):
                        description_parts.append(f"Size: {image_data['width']}x{image_data['height']}")
                    
                    description = " ".join(description_parts) if description_parts else "Document image"
                    
                    # Generate embedding for the description
                    embedding = self.generate_embedding(description)
                    
                    # Create point for Qdrant
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            'content_type': 'image',
                            'base64_image': base64_image,
                            'image_metadata': image_metadata,
                            'page': image_data.get('page', 0),
                            'image_index': image_data.get('index', 0),
                            'width': image_data.get('width', 0),
                            'height': image_data.get('height', 0),
                            'aspect_ratio': image_data.get('aspect_ratio', 0),
                            'size_bytes': image_data.get('size_bytes', 0),
                            'xref': image_data.get('xref', 0),
                            'description': description,
                            'source_file': image_data.get('source_file', ''),
                            'created_at': image_metadata.get('created_at', 0)
                        }
                    )
                    points.append(point)
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    errors.append(f"Image {image_data.get('file', 'unknown')}: {str(e)}")
                    print(f"Error processing image {image_data.get('file', 'unknown')}: {e}")
            
            # Upsert batch to Qdrant
            if points:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    print(f"Upserted batch of {len(points)} images to Qdrant")
                except Exception as e:
                    error_count += len(points)
                    errors.append(f"Batch upsert error: {str(e)}")
                    print(f"Error upserting image batch: {e}")
        
        return {
            "status": "completed",
            "processed": processed_count,
            "errors": error_count,
            "error_details": errors,
            "collection_name": self.collection_name
        }
    
    def search_similar(self, query: str, limit: int = 10, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        self._ensure_initialized()
        try:
            query_embedding = self.generate_embedding(query)
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for result in search_results:
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                })
            
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def search_images(self, query: str, limit: int = 10, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar images."""
        self._ensure_initialized()
        try:
            query_embedding = self.generate_embedding(query)
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter={
                    "must": [
                        {
                            "key": "content_type",
                            "match": {"value": "image"}
                        }
                    ]
                }
            )
            
            results = []
            for result in search_results:
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                })
            
            return results
        except Exception as e:
            print(f"Error searching images: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'name': collection_info.config.params.vectors.size,
                'vectors_count': collection_info.vectors_count,
                'points_count': collection_info.points_count,
                'status': collection_info.status
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
