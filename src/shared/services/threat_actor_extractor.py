#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threat Actor Extraction using spaCy NER
Extracts threat actors, organizations, and related entities from text content.
"""

import re
import spacy
from typing import Dict, List, Any
from datetime import datetime

class ThreatActorExtractor:
    """Extract threat actors and related entities using spaCy NER."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the threat actor extractor with lazy model loading.
        
        Args:
            model_name: spaCy model name (default: en_core_web_sm)
        """
        self.model_name = model_name
        self.nlp = None
        self._model_loaded = False
        
        # Threat actor related patterns and keywords
        self.threat_keywords = {
            'actor_types': [
                'threat actor', 'attacker', 'hacker', 'cybercriminal', 'malicious actor',
                'adversary', 'intruder', 'perpetrator', 'bad actor', 'threat group',
                'cyber group', 'hacking group', 'criminal organization', 'terrorist group',
                'state actor', 'nation state', 'apt', 'advanced persistent threat',
                'ransomware group', 'cyber gang', 'hacktivist', 'insider threat'
            ],
            'attack_indicators': [
                'attack', 'breach', 'compromise', 'infiltration', 'exploit', 'malware',
                'ransomware', 'phishing', 'social engineering', 'spear phishing',
                'watering hole', 'supply chain attack', 'zero day', 'vulnerability',
                'backdoor', 'trojan', 'botnet', 'command and control', 'c2'
            ],
            'target_keywords': [
                'target', 'victim', 'compromised', 'breached', 'infiltrated',
                'attacked', 'hacked', 'exploited', 'penetrated'
            ]
        }
        
        # Compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_model(self):
        """Load spaCy model lazily."""
        if self._model_loaded:
            return
            
        try:
            self.nlp = spacy.load(self.model_name)
            print(f"✅ Loaded spaCy model: {self.model_name}")
            self._model_loaded = True
        except OSError:
            print(f"❌ spaCy model '{self.model_name}' not found.")
            print("Please install it with: python -m spacy download en_core_web_sm")
            # Fallback to basic model
            try:
                self.nlp = spacy.blank("en")
                print("⚠️ Using blank spaCy model (limited NER capabilities)")
                self._model_loaded = True
            except Exception as e:
                print(f"❌ Failed to load spaCy: {e}")
                self.nlp = None
                self._model_loaded = False
    
    def _compile_patterns(self):
        """Compile regex patterns for threat actor detection."""
        self.threat_patterns = {
            'actor_types': re.compile(
                r'\b(?:' + '|'.join(re.escape(keyword) for keyword in self.threat_keywords['actor_types']) + r')\b',
                re.IGNORECASE
            ),
            'attack_indicators': re.compile(
                r'\b(?:' + '|'.join(re.escape(keyword) for keyword in self.threat_keywords['attack_indicators']) + r')\b',
                re.IGNORECASE
            ),
            'target_keywords': re.compile(
                r'\b(?:' + '|'.join(re.escape(keyword) for keyword in self.threat_keywords['target_keywords']) + r')\b',
                re.IGNORECASE
            )
        }
    
    def extract_threat_actors(self, text: str, page_number: int = None) -> Dict[str, Any]:
        """
        Extract threat actors and related entities from text.
        
        Args:
            text: Input text to analyze
            page_number: Page number for context
            
        Returns:
            Dictionary containing extracted threat actors and entities
        """
        # Load model lazily
        self._load_model()
        
        if not self.nlp or not text.strip():
            return {
                'threat_actors': [],
                'organizations': [],
                'persons': [],
                'locations': [],
                'attack_indicators': [],
                'summary': {
                    'total_threat_actors': 0,
                    'total_organizations': 0,
                    'total_persons': 0,
                    'total_locations': 0,
                    'total_attack_indicators': 0
                }
            }
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities by type
        threat_actors = []
        organizations = []
        persons = []
        locations = []
        attack_indicators = []
        
        # Process each entity
        for ent in doc.ents:
            entity_data = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0,  # spaCy doesn't provide confidence scores by default
                'page_number': page_number,
                'context': self._get_entity_context(text, ent.start_char, ent.end_char),
                'normalized': self._normalize_entity(ent.text, ent.label_)
            }
            
            # Categorize entities
            if ent.label_ in ['PERSON']:
                # Check if person might be a threat actor
                if self._is_potential_threat_actor(ent.text, text):
                    threat_actors.append(entity_data)
                else:
                    persons.append(entity_data)
            
            elif ent.label_ in ['ORG', 'GPE']:
                # Check if organization might be a threat actor
                if self._is_potential_threat_organization(ent.text, text):
                    threat_actors.append(entity_data)
                else:
                    organizations.append(entity_data)
            
            elif ent.label_ in ['LOC', 'GPE']:
                locations.append(entity_data)
            
            elif ent.label_ in ['MONEY', 'DATE', 'TIME']:
                # These might be relevant for attack context
                continue
        
        # Extract attack indicators using pattern matching
        attack_indicators = self._extract_attack_indicators(text, page_number)
        
        # Remove duplicates and merge similar entities
        threat_actors = self._deduplicate_entities(threat_actors)
        organizations = self._deduplicate_entities(organizations)
        persons = self._deduplicate_entities(persons)
        locations = self._deduplicate_entities(locations)
        
        return {
            'threat_actors': threat_actors,
            'organizations': organizations,
            'persons': persons,
            'locations': locations,
            'attack_indicators': attack_indicators,
            'summary': {
                'total_threat_actors': len(threat_actors),
                'total_organizations': len(organizations),
                'total_persons': len(persons),
                'total_locations': len(locations),
                'total_attack_indicators': len(attack_indicators)
            }
        }
    
    def _is_potential_threat_actor(self, entity_text: str, context: str) -> bool:
        """Check if a person entity might be a threat actor based on context."""
        entity_lower = entity_text.lower()
        
        # Check if entity appears in threat-related sentences
        sentences = context.split('.')
        for sentence in sentences:
            if entity_lower in sentence.lower():
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in self.threat_keywords['actor_types']):
                    return True
        
        return False
    
    def _is_potential_threat_organization(self, entity_text: str, context: str) -> bool:
        """Check if an organization entity might be a threat actor based on context."""
        context_lower = context.lower()
        
        # Known threat group patterns
        threat_group_patterns = [
            r'\b(?:apt|advanced persistent threat)\s+\d+',
            r'\b(?:group|gang|collective|organization)\b.*\b(?:hack|cyber|threat|malicious)\b',
            r'\b(?:lazarus|fancy bear|cozy bear|carbanak|fin7|maze|ryuk|conti)\b'
        ]
        
        # Check for threat group patterns
        for pattern in threat_group_patterns:
            if re.search(pattern, context_lower):
                return True
        
        # Check for threat-related context
        threat_context = any(
            keyword in context_lower 
            for keyword in self.threat_keywords['actor_types'] + self.threat_keywords['attack_indicators']
        )
        
        return threat_context
    
    def _extract_attack_indicators(self, text: str, page_number: int = None) -> List[Dict[str, Any]]:
        """Extract attack indicators using pattern matching."""
        indicators = []
        
        # Find attack indicator matches
        for match in self.threat_patterns['attack_indicators'].finditer(text):
            indicator_data = {
                'text': match.group(),
                'label': 'ATTACK_INDICATOR',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8,  # Pattern-based confidence
                'page_number': page_number,
                'context': self._get_entity_context(text, match.start(), match.end()),
                'normalized': match.group().lower()
            }
            indicators.append(indicator_data)
        
        return indicators
    
    def _get_entity_context(self, text: str, start: int, end: int, context_window: int = 100) -> str:
        """Get context around an entity."""
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        return text[context_start:context_end].strip()
    
    def _normalize_entity(self, entity_text: str, entity_label: str) -> str:
        """Normalize entity text for deduplication."""
        normalized = entity_text.strip().lower()
        
        # Remove common prefixes/suffixes
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        normalized = re.sub(r'\s+(inc|ltd|llc|corp|corporation|company|group|organization)$', '', normalized)
        
        return normalized
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on normalized text."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            normalized = entity['normalized']
            if normalized not in seen:
                seen.add(normalized)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_from_pdf_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract threat actors from a PDF processing manifest.
        
        Args:
            manifest: PDF processing manifest containing unified markdown
            
        Returns:
            Dictionary containing threat actors extracted from all pages
        """
        all_threat_actors = []
        all_organizations = []
        all_persons = []
        all_locations = []
        all_attack_indicators = []
        
        # Extract from unified markdown document
        unified_markdown = manifest.get('unified_markdown', {})
        if 'unified_document' in unified_markdown:
            try:
                with open(unified_markdown['unified_document'], 'r', encoding='utf-8') as f:
                    full_text = f.read()
                
                # Extract from full document
                result = self.extract_threat_actors(full_text)
                
                all_threat_actors.extend(result['threat_actors'])
                all_organizations.extend(result['organizations'])
                all_persons.extend(result['persons'])
                all_locations.extend(result['locations'])
                all_attack_indicators.extend(result['attack_indicators'])
                
            except Exception as e:
                print(f"Error reading unified document: {e}")
        
        # Also extract from individual pages if available
        if 'pages' in unified_markdown:
            for page_data in unified_markdown['pages']:
                page_file = page_data.get('file')
                page_number = page_data.get('page', 0)
                
                if page_file:
                    try:
                        with open(page_file, 'r', encoding='utf-8') as f:
                            page_text = f.read()
                        
                        result = self.extract_threat_actors(page_text, page_number)
                        
                        all_threat_actors.extend(result['threat_actors'])
                        all_organizations.extend(result['organizations'])
                        all_persons.extend(result['persons'])
                        all_locations.extend(result['locations'])
                        all_attack_indicators.extend(result['attack_indicators'])
                        
                    except Exception as e:
                        print(f"Error reading page {page_number}: {e}")
        
        # Final deduplication across all pages
        all_threat_actors = self._deduplicate_entities(all_threat_actors)
        all_organizations = self._deduplicate_entities(all_organizations)
        all_persons = self._deduplicate_entities(all_persons)
        all_locations = self._deduplicate_entities(all_locations)
        all_attack_indicators = self._deduplicate_entities(all_attack_indicators)
        
        return {
            'threat_actors': all_threat_actors,
            'organizations': all_organizations,
            'persons': all_persons,
            'locations': all_locations,
            'attack_indicators': all_attack_indicators,
            'summary': {
                'total_threat_actors': len(all_threat_actors),
                'total_organizations': len(all_organizations),
                'total_persons': len(all_persons),
                'total_locations': len(all_locations),
                'total_attack_indicators': len(all_attack_indicators)
            },
            'extraction_metadata': {
                'model_used': self.model_name,
                'extraction_timestamp': datetime.now().isoformat(),
                'total_pages_processed': len(unified_markdown.get('pages', []))
            }
        }
