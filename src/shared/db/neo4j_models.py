"""
Neo4j Knowledge Graph Models for Document Intelligence Pipeline

This module defines the knowledge graph schema with nodes and relationships
for capturing intelligence from processed documents.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import uuid


class IndicatorType(Enum):
    """Types of indicators that can be extracted from documents."""
    DOMAIN = "domain"
    URL = "url"
    IP_ADDRESS = "ip_address"
    EMAIL = "email"
    PHONE = "phone"
    SOCIAL_MEDIA = "social_media"
    TRACKING_ID = "tracking_id"
    HASH = "hash"
    FILE_PATH = "file_path"
    USERNAME = "username"


class ThreatLevel(Enum):
    """Threat level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CampaignType(Enum):
    """Types of threat campaigns."""
    PHISHING = "phishing"
    MALWARE = "malware"
    SOCIAL_ENGINEERING = "social_engineering"
    DATA_THEFT = "data_theft"
    RANSOMWARE = "ransomware"
    APT = "apt"  # Advanced Persistent Threat
    UNKNOWN = "unknown"


@dataclass
class BaseNode:
    """Base class for all knowledge graph nodes."""
    id: str
    created_at: datetime
    updated_at: datetime
    confidence_score: float = 0.0
    source: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j operations."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


@dataclass
class Document(BaseNode):
    """Document node representing a processed PDF or other document."""
    filename: str
    file_path: str
    file_hash: str
    file_size: int
    page_count: int
    language: str = "en"
    document_type: str = "pdf"
    processing_status: str = "completed"
    extracted_text_length: int = 0
    chunk_count: int = 0
    table_count: int = 0
    image_count: int = 0
    
    @classmethod
    def get_label(cls) -> str:
        return "Document"


@dataclass
class Indicator(BaseNode):
    """Indicator node representing extracted IOCs and other indicators."""
    value: str
    normalized_value: str
    indicator_type: IndicatorType
    context: str = ""
    page_number: Optional[int] = None
    chunk_id: Optional[str] = None
    extraction_method: str = "regex"
    threat_level: ThreatLevel = ThreatLevel.LOW
    is_verified: bool = False
    verification_date: Optional[datetime] = None
    verification_source: str = ""
    
    # Type-specific properties
    domain_info: Optional[Dict[str, Any]] = None
    ip_info: Optional[Dict[str, Any]] = None
    email_info: Optional[Dict[str, Any]] = None
    
    @classmethod
    def get_label(cls) -> str:
        return "Indicator"


@dataclass
class Campaign(BaseNode):
    """Campaign node representing threat campaigns or operations."""
    name: str
    campaign_type: CampaignType
    description: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: str = "active"  # active, inactive, completed
    attribution: str = ""
    target_sectors: List[str] = None
    target_countries: List[str] = None
    techniques_used: List[str] = None
    iocs_count: int = 0
    
    def __post_init__(self):
        if self.target_sectors is None:
            self.target_sectors = []
        if self.target_countries is None:
            self.target_countries = []
        if self.techniques_used is None:
            self.techniques_used = []
    
    @classmethod
    def get_label(cls) -> str:
        return "Campaign"


@dataclass
class ThreatActor(BaseNode):
    """Threat actor node representing individuals or groups."""
    name: str
    aliases: List[str] = None
    actor_type: str = "unknown"  # individual, group, state-sponsored, etc.
    country: str = ""
    description: str = ""
    motivation: str = ""
    capabilities: List[str] = None
    attribution_confidence: float = 0.0
    last_seen: Optional[datetime] = None
    campaigns_count: int = 0
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.capabilities is None:
            self.capabilities = []
    
    @classmethod
    def get_label(cls) -> str:
        return "ThreatActor"


@dataclass
class Chunk(BaseNode):
    """Text chunk node representing processed document segments."""
    content: str
    chunk_index: int
    content_type: str = "text"  # text, table, image
    page_number: int = 0
    chunk_size: int = 0
    embedding_id: Optional[str] = None  # Reference to vector store
    
    @classmethod
    def get_label(cls) -> str:
        return "Chunk"


# Relationship Types
class RelationshipType:
    """Constants for relationship types in the knowledge graph."""
    
    # Document relationships
    CONTAINS = "CONTAINS"
    EXTRACTED_FROM = "EXTRACTED_FROM"
    MENTIONED_IN = "MENTIONED_IN"
    
    # Indicator relationships
    RELATED_TO = "RELATED_TO"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"
    SIMILAR_TO = "SIMILAR_TO"
    RESOLVES_TO = "RESOLVES_TO"
    
    # Campaign relationships
    PART_OF_CAMPAIGN = "PART_OF_CAMPAIGN"
    USES_INDICATOR = "USES_INDICATOR"
    TARGETS = "TARGETS"
    ATTRIBUTED_TO = "ATTRIBUTED_TO"
    
    # Threat actor relationships
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    OPERATES = "OPERATES"
    COLLABORATES_WITH = "COLLABORATES_WITH"
    
    # Temporal relationships
    OCCURS_BEFORE = "OCCURS_BEFORE"
    OCCURS_AFTER = "OCCURS_AFTER"
    OCCURS_DURING = "OCCURS_DURING"


@dataclass
class Relationship:
    """Base class for knowledge graph relationships."""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = None
    confidence_score: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j operations."""
        data = {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat(),
            **self.properties
        }
        return data


# Specific relationship classes
@dataclass
class DocumentIndicatorRelationship(Relationship):
    """Relationship between document and indicator."""
    extraction_context: str = ""
    page_number: Optional[int] = None
    chunk_id: Optional[str] = None
    extraction_confidence: float = 0.0


@dataclass
class IndicatorIndicatorRelationship(Relationship):
    """Relationship between two indicators."""
    relationship_reason: str = ""
    co_occurrence_count: int = 1
    temporal_proximity: Optional[int] = None  # seconds


@dataclass
class CampaignIndicatorRelationship(Relationship):
    """Relationship between campaign and indicator."""
    usage_context: str = ""
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    frequency: int = 1


@dataclass
class CampaignThreatActorRelationship(Relationship):
    """Relationship between campaign and threat actor."""
    attribution_confidence: float = 0.0
    role: str = ""  # primary, secondary, suspected
    evidence: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.evidence is None:
            self.evidence = []


# Knowledge Graph Schema Definition
class KnowledgeGraphSchema:
    """Defines the complete knowledge graph schema."""
    
    NODE_LABELS = [
        Document.get_label(),
        Indicator.get_label(),
        Campaign.get_label(),
        ThreatActor.get_label(),
        Chunk.get_label()
    ]
    
    RELATIONSHIP_TYPES = [
        RelationshipType.CONTAINS,
        RelationshipType.EXTRACTED_FROM,
        RelationshipType.MENTIONED_IN,
        RelationshipType.RELATED_TO,
        RelationshipType.CO_OCCURS_WITH,
        RelationshipType.SIMILAR_TO,
        RelationshipType.RESOLVES_TO,
        RelationshipType.PART_OF_CAMPAIGN,
        RelationshipType.USES_INDICATOR,
        RelationshipType.TARGETS,
        RelationshipType.ATTRIBUTED_TO,
        RelationshipType.ASSOCIATED_WITH,
        RelationshipType.OPERATES,
        RelationshipType.COLLABORATES_WITH,
        RelationshipType.OCCURS_BEFORE,
        RelationshipType.OCCURS_AFTER,
        RelationshipType.OCCURS_DURING
    ]
    
    # Index definitions for better query performance
    INDEXES = {
        "Document": [
            "filename",
            "file_hash",
            "processing_status",
            "created_at"
        ],
        "Indicator": [
            "value",
            "normalized_value",
            "indicator_type",
            "threat_level",
            "is_verified"
        ],
        "Campaign": [
            "name",
            "campaign_type",
            "status",
            "attribution"
        ],
        "ThreatActor": [
            "name",
            "actor_type",
            "country"
        ],
        "Chunk": [
            "chunk_index",
            "content_type",
            "page_number"
        ]
    }
    
    # Constraint definitions
    CONSTRAINTS = {
        "Document": ["file_hash"],  # Unique file hash
        "Indicator": ["normalized_value"],  # Unique normalized indicator value
        "Campaign": ["name"],  # Unique campaign name
        "ThreatActor": ["name"]  # Unique threat actor name
    }


# Utility functions for creating nodes
def create_document_node(
    filename: str,
    file_path: str,
    file_hash: str,
    file_size: int,
    page_count: int,
    **kwargs
) -> Document:
    """Create a Document node with required fields."""
    return Document(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        updated_at=datetime.now(),
        filename=filename,
        file_path=file_path,
        file_hash=file_hash,
        file_size=file_size,
        page_count=page_count,
        **kwargs
    )


def create_indicator_node(
    value: str,
    normalized_value: str,
    indicator_type: IndicatorType,
    **kwargs
) -> Indicator:
    """Create an Indicator node with required fields."""
    return Indicator(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        updated_at=datetime.now(),
        value=value,
        normalized_value=normalized_value,
        indicator_type=indicator_type,
        **kwargs
    )


def create_campaign_node(
    name: str,
    campaign_type: CampaignType,
    **kwargs
) -> Campaign:
    """Create a Campaign node with required fields."""
    return Campaign(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        updated_at=datetime.now(),
        name=name,
        campaign_type=campaign_type,
        **kwargs
    )


def create_threat_actor_node(
    name: str,
    **kwargs
) -> ThreatActor:
    """Create a ThreatActor node with required fields."""
    return ThreatActor(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        updated_at=datetime.now(),
        name=name,
        **kwargs
    )


def create_chunk_node(
    content: str,
    chunk_index: int,
    **kwargs
) -> Chunk:
    """Create a Chunk node with required fields."""
    return Chunk(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        updated_at=datetime.now(),
        content=content,
        chunk_index=chunk_index,
        **kwargs
    )
