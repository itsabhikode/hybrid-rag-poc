"""
Simplified PDF Parser Service using PyMuPDF4LLM
Handles text, tables, and images extraction using only PyMuPDF4LLM
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import base64

import fitz  # PyMuPDF
import pymupdf4llm

logger = logging.getLogger(__name__)


class EnhancedPDFParser:
    """
    Simplified PDF parser using PyMuPDF4LLM that handles:
    - Text extraction with layout awareness
    - Table extraction and conversion to markdown/CSV
    - Image extraction
    - Structured content output
    """
    
    def __init__(self):
        """Initialize the PDF parser"""
        self.supported_formats = ['.pdf']
    
    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a PDF file using PyMuPDF4LLM
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.suffix.lower() != '.pdf':
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Initialize result structure
            result = {
                "is_processed": True,
                "text": "",
                "structured_content": {
                    "pages": [],
                    "tables": [],
                    "images": [],
                    "metadata": {}
                },
                "metadata": {
                    "file_type": "pdf",
                    "file_size": file_path.stat().st_size,
                    "total_pages": 0,
                    "extraction_methods": ["pymupdf4llm"]
                },
                "error": ""
            }
            
            # Extract using PyMuPDF4LLM
            self._extract_with_pymupdf4llm(file_path, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            return {
                "is_processed": False,
                "text": "",
                "structured_content": {"pages": [], "tables": [], "images": []},
                "metadata": {"file_type": "pdf", "file_size": 0},
                "error": str(e)
            }
    
    def _extract_with_pymupdf4llm(self, file_path: Path, result: Dict[str, Any]) -> None:
        """Extract text, tables, and images using PyMuPDF4LLM"""
        try:
            # Use PyMuPDF4LLM to extract structured content
            md_text = pymupdf4llm.to_markdown(str(file_path))
            
            # Open the document for additional processing
            doc = fitz.open(file_path)
            result["metadata"]["total_pages"] = len(doc)
            
            # Extract full text
            full_text = ""
            page_contents = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text from page
                page_text = page.get_text()
                full_text += page_text + "\n"
                
                # Extract images from page
                page_images = self._extract_images_from_page(page, page_num, result)
                
                # Extract tables using PyMuPDF4LLM table detection
                page_tables = self._extract_tables_from_page(page, page_num, result)
                
                page_content = {
                    "page_number": page_num + 1,
                    "text": page_text,
                    "images": page_images,
                    "tables": page_tables
                }
                
                page_contents.append(page_content)
                result["structured_content"]["pages"].append(page_content)
            
            # Set the full text
            result["text"] = full_text.strip()
            result["structured_content"]["metadata"] = {
                "markdown_text": md_text,
                "total_pages": len(doc),
                "total_images": len(result["structured_content"]["images"]),
                "total_tables": len(result["structured_content"]["tables"])
            }
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF4LLM extraction failed: {str(e)}")
            raise
    
    def _extract_images_from_page(self, page, page_num: int, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract images from a page using PyMuPDF"""
        try:
            images = []
            
            # Get image list from the page
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                # Get image data
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # Convert to base64 if it's not an image mask
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    img_base64 = base64.b64encode(img_data).decode()
                    
                    # Get image position (approximate)
                    img_rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
                    
                    image_info = {
                        "page_number": page_num + 1,
                        "image_index": img_idx,
                        "xref": xref,
                        "width": pix.width,
                        "height": pix.height,
                        "colorspace": pix.colorspace.name if pix.colorspace else "unknown",
                        "bbox": list(img_rect) if img_rect else None,
                        "image_base64": f"data:image/png;base64,{img_base64}"
                    }
                    
                    images.append(image_info)
                    result["structured_content"]["images"].append(image_info)
                
                pix = None  # Free memory
            
            return images
            
        except Exception as e:
            logger.error(f"Image extraction from page failed: {str(e)}")
            return []
    
    def _extract_tables_from_page(self, page, page_num: int, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tables from a page using PyMuPDF table detection"""
        try:
            tables = []
            
            # Find tables using PyMuPDF's table detection
            table_dicts = page.find_tables()
            
            for table_idx, table_dict in enumerate(table_dicts):
                try:
                    # Extract table data
                    table_data = table_dict.extract()
                    
                    if table_data and len(table_data) > 0:
                        table_info = {
                            "page_number": page_num + 1,
                            "table_index": table_idx,
                            "data": table_data,
                            "rows": len(table_data),
                            "columns": len(table_data[0]) if table_data else 0,
                            "bbox": list(table_dict.bbox) if hasattr(table_dict, 'bbox') else None
                        }
                        
                        tables.append(table_info)
                        result["structured_content"]["tables"].append(table_info)
                        
                except Exception as table_error:
                    logger.warning(f"Failed to extract table {table_idx} from page {page_num + 1}: {table_error}")
                    continue
            
            return tables
            
        except Exception as e:
            logger.error(f"Table extraction from page failed: {str(e)}")
            return []


def parse_pdf_enhanced(file_path: str) -> Dict[str, Any]:
    """
    Enhanced PDF parsing function using PyMuPDF4LLM
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary containing extracted content and metadata
    """
    parser = EnhancedPDFParser()
    return parser.parse_pdf(file_path)