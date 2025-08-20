"""
Centralized data management system for rate cards and site lists.
"""

import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib

from .parsers import RateCardParser, SiteListParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataCacheEntry:
    """Represents a cached data entry with metadata."""
    data: Dict[str, Any]
    file_path: str
    file_hash: str
    last_updated: datetime
    last_accessed: datetime


class DataManager:
    """
    Centralized data access and management system.
    
    Provides caching, validation, and unified access to rate cards and site lists.
    """
    
    def __init__(self, cache_dir: str = ".cache", cache_ttl_hours: int = 24):
        """
        Initialize the DataManager.
        
        Args:
            cache_dir: Directory to store cached data
            cache_ttl_hours: Time-to-live for cached data in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # In-memory cache
        self._rate_card_cache: Optional[DataCacheEntry] = None
        self._site_list_cache: Optional[DataCacheEntry] = None
        
        # Default file paths
        self.default_rate_card_path = "adzymic_rate_card.xlsx"
        self.default_site_list_path = "APX Sitelist - Regional.xlsx"
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of a file for change detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash string
        """
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5()
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
                return file_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash for {file_path}: {str(e)}")
            return ""
    
    def _save_cache_to_disk(self, cache_entry: DataCacheEntry, cache_type: str):
        """
        Save cache entry to disk.
        
        Args:
            cache_entry: Cache entry to save
            cache_type: Type of cache ('rate_card' or 'site_list')
        """
        try:
            cache_file = self.cache_dir / f"{cache_type}_cache.json"
            
            # Convert datetime objects to ISO format for JSON serialization
            cache_data = {
                'data': cache_entry.data,
                'file_path': cache_entry.file_path,
                'file_hash': cache_entry.file_hash,
                'last_updated': cache_entry.last_updated.isoformat(),
                'last_accessed': cache_entry.last_accessed.isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
                
            logger.info(f"Saved {cache_type} cache to disk")
            
        except Exception as e:
            logger.error(f"Error saving {cache_type} cache to disk: {str(e)}")
    
    def _load_cache_from_disk(self, cache_type: str) -> Optional[DataCacheEntry]:
        """
        Load cache entry from disk.
        
        Args:
            cache_type: Type of cache ('rate_card' or 'site_list')
            
        Returns:
            Cache entry if found and valid, None otherwise
        """
        try:
            cache_file = self.cache_dir / f"{cache_type}_cache.json"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Convert ISO format back to datetime objects
            cache_entry = DataCacheEntry(
                data=cache_data['data'],
                file_path=cache_data['file_path'],
                file_hash=cache_data['file_hash'],
                last_updated=datetime.fromisoformat(cache_data['last_updated']),
                last_accessed=datetime.fromisoformat(cache_data['last_accessed'])
            )
            
            logger.info(f"Loaded {cache_type} cache from disk")
            return cache_entry
            
        except Exception as e:
            logger.error(f"Error loading {cache_type} cache from disk: {str(e)}")
            return None
    
    def _is_cache_valid(self, cache_entry: DataCacheEntry) -> bool:
        """
        Check if cache entry is still valid.
        
        Args:
            cache_entry: Cache entry to validate
            
        Returns:
            True if cache is valid, False otherwise
        """
        try:
            # Check if file still exists
            if not os.path.exists(cache_entry.file_path):
                logger.warning(f"Cached file no longer exists: {cache_entry.file_path}")
                return False
            
            # Check if file has been modified
            current_hash = self._get_file_hash(cache_entry.file_path)
            if current_hash != cache_entry.file_hash:
                logger.info(f"File has been modified: {cache_entry.file_path}")
                return False
            
            # Check if cache has expired
            if datetime.now() - cache_entry.last_updated > self.cache_ttl:
                logger.info(f"Cache has expired for: {cache_entry.file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating cache: {str(e)}")
            return False
    
    def load_rate_cards(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and cache rate card data.
        
        Args:
            file_path: Path to rate card Excel file. Uses default if None.
            
        Returns:
            Rate card data dictionary
            
        Raises:
            FileNotFoundError: If rate card file is not found
            ValueError: If rate card data is invalid
        """
        if file_path is None:
            file_path = self.default_rate_card_path
        
        # Check in-memory cache first
        if (self._rate_card_cache and 
            self._rate_card_cache.file_path == file_path and 
            self._is_cache_valid(self._rate_card_cache)):
            
            # Update access time
            self._rate_card_cache.last_accessed = datetime.now()
            logger.info("Using in-memory rate card cache")
            return self._rate_card_cache.data
        
        # Check disk cache
        disk_cache = self._load_cache_from_disk('rate_card')
        if (disk_cache and 
            disk_cache.file_path == file_path and 
            self._is_cache_valid(disk_cache)):
            
            # Load into memory cache
            disk_cache.last_accessed = datetime.now()
            self._rate_card_cache = disk_cache
            logger.info("Using disk rate card cache")
            return disk_cache.data
        
        # Parse fresh data
        logger.info(f"Parsing rate card from: {file_path}")
        parser = RateCardParser(file_path)
        data = parser.parse_all_rates()
        
        # Create cache entry
        file_hash = self._get_file_hash(file_path)
        cache_entry = DataCacheEntry(
            data=data,
            file_path=file_path,
            file_hash=file_hash,
            last_updated=datetime.now(),
            last_accessed=datetime.now()
        )
        
        # Store in memory and disk
        self._rate_card_cache = cache_entry
        self._save_cache_to_disk(cache_entry, 'rate_card')
        
        logger.info("Rate card data loaded and cached successfully")
        return data
    
    def load_site_lists(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and cache site list data.
        
        Args:
            file_path: Path to site list Excel file. Uses default if None.
            
        Returns:
            Site list data dictionary
            
        Raises:
            FileNotFoundError: If site list file is not found
            ValueError: If site list data is invalid
        """
        if file_path is None:
            file_path = self.default_site_list_path
        
        # Check in-memory cache first
        if (self._site_list_cache and 
            self._site_list_cache.file_path == file_path and 
            self._is_cache_valid(self._site_list_cache)):
            
            # Update access time
            self._site_list_cache.last_accessed = datetime.now()
            logger.info("Using in-memory site list cache")
            return self._site_list_cache.data
        
        # Check disk cache
        disk_cache = self._load_cache_from_disk('site_list')
        if (disk_cache and 
            disk_cache.file_path == file_path and 
            self._is_cache_valid(disk_cache)):
            
            # Load into memory cache
            disk_cache.last_accessed = datetime.now()
            self._site_list_cache = disk_cache
            logger.info("Using disk site list cache")
            return disk_cache.data
        
        # Parse fresh data
        logger.info(f"Parsing site list from: {file_path}")
        parser = SiteListParser(file_path)
        data = parser.parse_all_markets()
        
        # Create cache entry
        file_hash = self._get_file_hash(file_path)
        cache_entry = DataCacheEntry(
            data=data,
            file_path=file_path,
            file_hash=file_hash,
            last_updated=datetime.now(),
            last_accessed=datetime.now()
        )
        
        # Store in memory and disk
        self._site_list_cache = cache_entry
        self._save_cache_to_disk(cache_entry, 'site_list')
        
        logger.info("Site list data loaded and cached successfully")
        return data
    
    def get_market_data(self, market: str) -> Dict[str, Any]:
        """
        Get comprehensive data for a specific market.
        
        Args:
            market: Market code (e.g., 'SG', 'MY')
            
        Returns:
            Dictionary containing rate card and site data for the market
        """
        try:
            # Load both data sources
            rate_data = self.load_rate_cards()
            site_data = self.load_site_lists()
            
            # Extract market-specific data
            market_info = {
                'market_code': market,
                'available': False,
                'rate_card': {
                    'impact_formats': {},
                    'reach_formats': {}
                },
                'sites': {
                    'total_sites': 0,
                    'formats': [],
                    'categories': [],
                    'sites_by_format': {},
                    'sites_by_category': {}
                }
            }
            
            # Check if market exists in rate card
            if market in rate_data.get('markets', []):
                market_info['available'] = True
                
                # Extract impact rates for this market
                for format_name, format_rates in rate_data.get('impact', {}).items():
                    if market in format_rates:
                        market_info['rate_card']['impact_formats'][format_name] = format_rates[market]
                
                # Extract reach rates for this market
                for format_name, format_rates in rate_data.get('reach', {}).items():
                    if market in format_rates:
                        market_info['rate_card']['reach_formats'][format_name] = format_rates[market]
            
            # Check if market exists in site list
            if market in site_data.get('markets', {}):
                market_sites = site_data['markets'][market]
                market_info['sites'] = {
                    'total_sites': len(market_sites.get('all_sites', [])),
                    'formats': market_sites.get('formats', []),
                    'categories': market_sites.get('categories', []),
                    'sites_by_format': market_sites.get('sites_by_format', {}),
                    'sites_by_category': market_sites.get('sites_by_category', {})
                }
                
                if not market_info['available']:
                    market_info['available'] = True
            
            return market_info
            
        except Exception as e:
            logger.error(f"Error getting market data for {market}: {str(e)}")
            raise
    
    def validate_data_freshness(self) -> Dict[str, Any]:
        """
        Validate freshness and completeness of cached data.
        
        Returns:
            Dictionary containing validation results and recommendations
        """
        validation_result = {
            'rate_card': {
                'status': 'unknown',
                'last_updated': None,
                'file_exists': False,
                'cache_valid': False,
                'recommendations': []
            },
            'site_list': {
                'status': 'unknown',
                'last_updated': None,
                'file_exists': False,
                'cache_valid': False,
                'recommendations': []
            },
            'overall_status': 'unknown'
        }
        
        try:
            # Check rate card
            if os.path.exists(self.default_rate_card_path):
                validation_result['rate_card']['file_exists'] = True
                
                if self._rate_card_cache:
                    validation_result['rate_card']['last_updated'] = self._rate_card_cache.last_updated.isoformat()
                    validation_result['rate_card']['cache_valid'] = self._is_cache_valid(self._rate_card_cache)
                    
                    if validation_result['rate_card']['cache_valid']:
                        validation_result['rate_card']['status'] = 'fresh'
                    else:
                        validation_result['rate_card']['status'] = 'stale'
                        validation_result['rate_card']['recommendations'].append('Reload rate card data')
                else:
                    validation_result['rate_card']['status'] = 'not_loaded'
                    validation_result['rate_card']['recommendations'].append('Load rate card data')
            else:
                validation_result['rate_card']['status'] = 'missing'
                validation_result['rate_card']['recommendations'].append('Upload rate card file')
            
            # Check site list
            if os.path.exists(self.default_site_list_path):
                validation_result['site_list']['file_exists'] = True
                
                if self._site_list_cache:
                    validation_result['site_list']['last_updated'] = self._site_list_cache.last_updated.isoformat()
                    validation_result['site_list']['cache_valid'] = self._is_cache_valid(self._site_list_cache)
                    
                    if validation_result['site_list']['cache_valid']:
                        validation_result['site_list']['status'] = 'fresh'
                    else:
                        validation_result['site_list']['status'] = 'stale'
                        validation_result['site_list']['recommendations'].append('Reload site list data')
                else:
                    validation_result['site_list']['status'] = 'not_loaded'
                    validation_result['site_list']['recommendations'].append('Load site list data')
            else:
                validation_result['site_list']['status'] = 'missing'
                validation_result['site_list']['recommendations'].append('Upload site list file')
            
            # Determine overall status
            rate_status = validation_result['rate_card']['status']
            site_status = validation_result['site_list']['status']
            
            if rate_status == 'fresh' and site_status == 'fresh':
                validation_result['overall_status'] = 'ready'
            elif rate_status in ['fresh', 'stale'] and site_status in ['fresh', 'stale']:
                validation_result['overall_status'] = 'needs_refresh'
            else:
                validation_result['overall_status'] = 'incomplete'
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating data freshness: {str(e)}")
            validation_result['overall_status'] = 'error'
            return validation_result
    
    def get_available_markets(self) -> List[str]:
        """
        Get list of all available markets from both data sources.
        
        Returns:
            Sorted list of unique market codes
        """
        try:
            markets = set()
            
            # Get markets from rate card
            try:
                rate_data = self.load_rate_cards()
                markets.update(rate_data.get('markets', []))
            except Exception as e:
                logger.warning(f"Could not load rate card markets: {str(e)}")
            
            # Get markets from site list
            try:
                site_data = self.load_site_lists()
                markets.update(site_data.get('available_markets', []))
            except Exception as e:
                logger.warning(f"Could not load site list markets: {str(e)}")
            
            return sorted(list(markets))
            
        except Exception as e:
            logger.error(f"Error getting available markets: {str(e)}")
            return []
    
    def get_format_rate(self, format_name: str, market: str, product_type: str = 'impact') -> Optional[float]:
        """
        Get rate for a specific format in a market.
        
        Args:
            format_name: Name of the ad format
            market: Market code
            product_type: 'impact' or 'reach'
            
        Returns:
            Rate as float, or None if not found
        """
        try:
            rate_data = self.load_rate_cards()
            
            if product_type not in rate_data:
                return None
            
            format_rates = rate_data[product_type].get(format_name, {})
            return format_rates.get(market)
            
        except Exception as e:
            logger.error(f"Error getting format rate: {str(e)}")
            return None
    
    def get_sites_for_format(self, market: str, format_name: str) -> List[Dict[str, Any]]:
        """
        Get sites that support a specific format in a market.
        
        Args:
            market: Market code
            format_name: Ad format name
            
        Returns:
            List of site information dictionaries
        """
        try:
            site_data = self.load_site_lists()
            
            if market not in site_data.get('markets', {}):
                return []
            
            market_sites = site_data['markets'][market]
            return market_sites.get('sites_by_format', {}).get(format_name, [])
            
        except Exception as e:
            logger.error(f"Error getting sites for format: {str(e)}")
            return []
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            self._rate_card_cache = None
            self._site_list_cache = None
            
            # Remove cache files
            for cache_file in self.cache_dir.glob("*_cache.json"):
                cache_file.unlink()
            
            logger.info("All caches cleared")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached data.
        
        Returns:
            Dictionary containing cache statistics
        """
        stats = {
            'rate_card': {
                'in_memory': self._rate_card_cache is not None,
                'last_accessed': None,
                'file_path': None
            },
            'site_list': {
                'in_memory': self._site_list_cache is not None,
                'last_accessed': None,
                'file_path': None
            },
            'cache_dir_size': 0
        }
        
        try:
            if self._rate_card_cache:
                stats['rate_card']['last_accessed'] = self._rate_card_cache.last_accessed.isoformat()
                stats['rate_card']['file_path'] = self._rate_card_cache.file_path
            
            if self._site_list_cache:
                stats['site_list']['last_accessed'] = self._site_list_cache.last_accessed.isoformat()
                stats['site_list']['file_path'] = self._site_list_cache.file_path
            
            # Calculate cache directory size
            if self.cache_dir.exists():
                stats['cache_dir_size'] = sum(f.stat().st_size for f in self.cache_dir.glob("*") if f.is_file())
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
        
        return stats