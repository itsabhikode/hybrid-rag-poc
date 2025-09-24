import re
from typing import Dict, List, Any
import hashlib
from datetime import datetime

class TextChunker:
    """Handle markdown chunking for unified collection."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 250, 
                 min_chunk_size: int = 100, max_chunk_size: int = 2000):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    
    def chunk_markdown(self, markdown_text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk markdown content that may contain both text and tables."""
        if not markdown_text or len(markdown_text.strip()) < self.min_chunk_size:
            return []
        
        chunks = []
        markdown_text = markdown_text.strip()
        
        # Split by sections (headers, paragraphs, tables) for better chunk boundaries
        sections = self._split_markdown_sections(markdown_text)
        
        current_chunk = ""
        chunk_index = 0
        
        for section in sections:
            # If adding this section would exceed chunk size, save current chunk
            if current_chunk and len(current_chunk + "\n\n" + section) > self.chunk_size:
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        current_chunk.strip(), 
                        chunk_index, 
                        metadata or {}
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + "\n\n" + section if overlap_text else section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                current_chunk.strip(), 
                chunk_index, 
                metadata or {}
            ))
        
        return chunks

    
    def _create_chunk(self, content: str, chunk_index: int, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chunk with metadata."""
        chunk_id = hashlib.md5(f"{content}_{chunk_index}_{base_metadata.get('page', '')}".encode()).hexdigest()
        
        return {
            'id': chunk_id,
            'content': content,
            'chunk_index': chunk_index,
            'content_type': base_metadata.get('content_type', 'text'),
            'page': base_metadata.get('page', 0),
            'source_file': base_metadata.get('source_file', ''),
            'chunk_size': len(content),
            'created_at': datetime.now().isoformat(),
            'metadata': {
                **base_metadata,
                'chunk_size': len(content),
                'chunk_index': chunk_index
            }
        }
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        if not text or self.chunk_overlap <= 0:
            return ""
        
        # Split by sentences and take last few sentences for overlap
        sentences = re.split(r'(?<=[.!?])\s+', text)
        overlap_sentences = []
        overlap_length = 0
        
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break
        
        return " ".join(overlap_sentences)
    
    def _split_markdown_sections(self, markdown_text: str) -> List[str]:
        """Split markdown text into logical sections (headers, paragraphs, tables)."""
        sections = []
        lines = markdown_text.split('\n')
        current_section = []
        
        for line in lines:
            line = line.strip()
            
            # Check if this is a header (starts with #)
            if line.startswith('#'):
                # Save current section if it exists
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
                # Add header as its own section
                sections.append(line)
            
            # Check if this is a table row (contains |)
            elif '|' in line and line.count('|') >= 2:
                # If we're not already in a table section, start one
                if current_section and not any('|' in line_item for line_item in current_section):
                    # Save current non-table section
                    sections.append('\n'.join(current_section))
                    current_section = []
                current_section.append(line)
            
            # Check if this is a separator line (table separator)
            elif re.match(r'^[\|\-\s:]+$', line) and '|' in line:
                current_section.append(line)
            
            # Regular text line
            else:
                # If we were in a table section, save it
                if current_section and any('|' in line_item for line_item in current_section):
                    sections.append('\n'.join(current_section))
                    current_section = []
                
                if line:  # Only add non-empty lines
                    current_section.append(line)
                elif current_section:  # Empty line after content - end current section
                    sections.append('\n'.join(current_section))
                    current_section = []
        
        # Add final section
        if current_section:
            sections.append('\n'.join(current_section))
        
        # Filter out empty sections
        return [section for section in sections if section.strip()]

