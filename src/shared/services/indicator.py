import re
import ipaddress
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional


class IndicatorExtractor:
    """Extract and normalize various types of indicators from text."""
    
    def __init__(self):
        self.patterns = self._compile_patterns()
        self.social_platforms = {
            'facebook': r'facebook\.com/[^/\s]+',
            'twitter': r'(?:twitter\.com|x\.com)/[^/\s]+',
            'instagram': r'instagram\.com/[^/\s]+',
            'youtube': r'(?:youtube\.com|youtu\.be)/[^/\s]+',
            'linkedin': r'linkedin\.com/in/[^/\s]+',
            'tiktok': r'tiktok\.com/@[^/\s]+',
            'telegram': r't\.me/[^/\s]+',
            'reddit': r'reddit\.com/r/[^/\s]+',
            'vk': r'vk\.com/[^/\s]+',
            'truth_social': r'truthsocial\.com/@[^/\s]+',
            'parler': r'parler\.com/[^/\s]+'
        }
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for all indicator types."""
        return {
            # Domains (e.g., example.com)
            'domain': re.compile(
                r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b',
                re.IGNORECASE
            ),
            
            # URLs (https://...)
            'url': re.compile(
                r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
                re.IGNORECASE
            ),
            
            # IP Addresses (IPv4 and IPv6)
            'ipv4': re.compile(
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
            ),
            'ipv6': re.compile(
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|'
                r'\b(?:[0-9a-fA-F]{1,4}:)*::(?:[0-9a-fA-F]{1,4}:)*[0-9a-fA-F]{1,4}\b'
            ),
            
            # Phone Numbers (various formats)
            'phone': re.compile(
                r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})|'
                r'(?:\+?[1-9]\d{1,14})|'
                r'(?:\(\d{3}\)\s?\d{3}-\d{4})|'
                r'(?:\d{3}-\d{3}-\d{4})|'
                r'(?:\d{3}\.\d{3}\.\d{4})|'
                r'(?:\d{3}\s\d{3}\s\d{4})',
                re.IGNORECASE
            ),
            
            # Email Addresses
            'email': re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            
            # Google Analytics IDs
            'ga_id': re.compile(
                r'\bUA-\d{4,10}-\d{1,4}\b',
                re.IGNORECASE
            ),
            
            # Google AdSense IDs
            'adsense_id': re.compile(
                r'\bpub-\d{16}\b',
                re.IGNORECASE
            ),
            
            # Facebook Pixel IDs
            'fb_pixel': re.compile(
                r'\b\d{15,16}\b(?=.*facebook|.*fb)',
                re.IGNORECASE
            )
        }
    
    def extract_indicators(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all types of indicators from text."""
        indicators = {
            'domains': [],
            'urls': [],
            'ip_addresses': [],
            'phone_numbers': [],
            'emails': [],
            'social_media': [],
            'tracking_ids': []
        }
        
        # Extract basic indicators
        indicators['domains'] = self._extract_domains(text)
        indicators['urls'] = self._extract_urls(text)
        indicators['ip_addresses'] = self._extract_ip_addresses(text)
        indicators['phone_numbers'] = self._extract_phone_numbers(text)
        indicators['emails'] = self._extract_emails(text)
        indicators['social_media'] = self._extract_social_media(text)
        indicators['tracking_ids'] = self._extract_tracking_ids(text)
        
        return indicators
    
    def _extract_domains(self, text: str) -> List[Dict[str, Any]]:
        """Extract and normalize domains."""
        domains = []
        matches = self.patterns['domain'].findall(text)
        
        for match in matches:
            domain = match.lower().strip()
            # Filter out common false positives
            if not self._is_valid_domain(domain):
                continue
            
            domains.append({
                'value': domain,
                'normalized': domain,
                'type': 'domain',
                'subdomain': self._extract_subdomain(domain),
                'root_domain': self._extract_root_domain(domain)
            })
        
        return list({d['normalized']: d for d in domains}.values())  # Remove duplicates
    
    def _extract_urls(self, text: str) -> List[Dict[str, Any]]:
        """Extract and normalize URLs."""
        urls = []
        matches = self.patterns['url'].findall(text)
        
        for match in matches:
            url = match.strip()
            try:
                parsed = urlparse(url)
                if parsed.netloc:
                    urls.append({
                        'value': url,
                        'normalized': self._normalize_url(url),
                        'type': 'url',
                        'domain': parsed.netloc.lower(),
                        'path': parsed.path,
                        'query': parsed.query,
                        'fragment': parsed.fragment,
                        'scheme': parsed.scheme
                    })
            except Exception:
                continue
        
        return list({u['normalized']: u for u in urls}.values())  # Remove duplicates
    
    def _extract_ip_addresses(self, text: str) -> List[Dict[str, Any]]:
        """Extract and normalize IP addresses."""
        ips = []
        
        # IPv4
        for match in self.patterns['ipv4'].findall(text):
            try:
                ip = ipaddress.IPv4Address(match)
                ips.append({
                    'value': match,
                    'normalized': str(ip),
                    'type': 'ipv4',
                    'version': 4,
                    'is_private': ip.is_private,
                    'is_loopback': ip.is_loopback,
                    'is_multicast': ip.is_multicast
                })
            except ipaddress.AddressValueError:
                continue
        
        # IPv6
        for match in self.patterns['ipv6'].findall(text):
            try:
                ip = ipaddress.IPv6Address(match)
                ips.append({
                    'value': match,
                    'normalized': str(ip),
                    'type': 'ipv6',
                    'version': 6,
                    'is_private': ip.is_private,
                    'is_loopback': ip.is_loopback,
                    'is_multicast': ip.is_multicast
                })
            except ipaddress.AddressValueError:
                continue
        
        return list({ip['normalized']: ip for ip in ips}.values())  # Remove duplicates
    
    def _extract_phone_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract and normalize phone numbers."""
        phones = []
        matches = self.patterns['phone'].findall(text)
        
        for match in matches:
            if isinstance(match, tuple):
                # Handle grouped matches
                phone = ''.join(match)
            else:
                phone = match
            
            normalized = self._normalize_phone(phone)
            if normalized:
                phones.append({
                    'value': phone,
                    'normalized': normalized,
                    'type': 'phone',
                    'country_code': self._extract_country_code(normalized),
                    'formatted': self._format_phone(normalized)
                })
        
        return list({p['normalized']: p for p in phones}.values())  # Remove duplicates
    
    def _extract_emails(self, text: str) -> List[Dict[str, Any]]:
        """Extract and normalize email addresses."""
        emails = []
        matches = self.patterns['email'].findall(text)
        
        for match in matches:
            email = match.lower().strip()
            emails.append({
                'value': email,
                'normalized': email,
                'type': 'email',
                'local_part': email.split('@')[0],
                'domain': email.split('@')[1] if '@' in email else ''
            })
        
        return list({e['normalized']: e for e in emails}.values())  # Remove duplicates
    
    def _extract_social_media(self, text: str) -> List[Dict[str, Any]]:
        """Extract social media handles and URLs."""
        social_indicators = []
        
        for platform, pattern in self.social_platforms.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                social_indicators.append({
                    'value': match,
                    'normalized': match.lower(),
                    'type': 'social_media',
                    'platform': platform,
                    'handle': self._extract_handle(match, platform)
                })
        
        return list({s['normalized']: s for s in social_indicators}.values())  # Remove duplicates
    
    def _extract_tracking_ids(self, text: str) -> List[Dict[str, Any]]:
        """Extract tracking IDs."""
        tracking_ids = []
        
        # Google Analytics
        for match in self.patterns['ga_id'].findall(text):
            tracking_ids.append({
                'value': match,
                'normalized': match.upper(),
                'type': 'tracking_id',
                'service': 'google_analytics',
                'account_id': match.split('-')[1] if '-' in match else '',
                'property_id': match.split('-')[2] if len(match.split('-')) > 2 else ''
            })
        
        # Google AdSense
        for match in self.patterns['adsense_id'].findall(text):
            tracking_ids.append({
                'value': match,
                'normalized': match.upper(),
                'type': 'tracking_id',
                'service': 'google_adsense'
            })
        
        # Facebook Pixel
        for match in self.patterns['fb_pixel'].findall(text):
            tracking_ids.append({
                'value': match,
                'normalized': match,
                'type': 'tracking_id',
                'service': 'facebook_pixel'
            })
        
        return list({t['normalized']: t for t in tracking_ids}.values())  # Remove duplicates
    
    # Helper methods
    def _is_valid_domain(self, domain: str) -> bool:
        """Check if domain is valid and not a common false positive."""
        invalid_domains = {
            'localhost', 'example.com', 'example.org', 'test.com',
            'domain.com', 'website.com', 'site.com'
        }
        
        if domain in invalid_domains:
            return False
        
        # Check for valid TLD
        if '.' not in domain or len(domain.split('.')[-1]) < 2:
            return False
        
        return True
    
    def _extract_subdomain(self, domain: str) -> Optional[str]:
        """Extract subdomain from domain."""
        parts = domain.split('.')
        if len(parts) > 2:
            return '.'.join(parts[:-2])
        return None
    
    def _extract_root_domain(self, domain: str) -> str:
        """Extract root domain from domain."""
        parts = domain.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
        return domain
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistent comparison."""
        try:
            parsed = urlparse(url)
            normalized = f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path}"
            if parsed.query:
                normalized += f"?{parsed.query}"
            if parsed.fragment:
                normalized += f"#{parsed.fragment}"
            return normalized
        except Exception:
            return url.lower()
    
    def _normalize_phone(self, phone: str) -> Optional[str]:
        """Normalize phone number to digits only."""
        digits = re.sub(r'[^\d]', '', phone)
        if len(digits) >= 10:
            return digits
        return None
    
    def _extract_country_code(self, phone: str) -> Optional[str]:
        """Extract country code from phone number."""
        if len(phone) > 10:
            return phone[:-10]
        return None
    
    def _format_phone(self, phone: str) -> str:
        """Format phone number for display."""
        if len(phone) == 10:
            return f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
        elif len(phone) == 11 and phone.startswith('1'):
            return f"+1 ({phone[1:4]}) {phone[4:7]}-{phone[7:]}"
        return phone
    
    def _extract_handle(self, url: str, platform: str) -> Optional[str]:
        """Extract handle/username from social media URL."""
        if platform == 'telegram':
            return url.split('t.me/')[-1] if 't.me/' in url else None
        elif platform == 'tiktok':
            return url.split('@')[-1] if '@' in url else None
        elif platform == 'truth_social':
            return url.split('@')[-1] if '@' in url else None
        else:
            return url.split('/')[-1] if '/' in url else None
