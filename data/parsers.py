"""
Data parsers for Excel files containing rate cards and site lists.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateCardParser:
    """
    Parser for Excel rate card files containing APX Impact and Reach pricing data.
    
    Handles various Excel file formats and provides data validation and error handling
    for malformed files.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the parser with a rate card file path.
        
        Args:
            file_path: Path to the Excel rate card file
        """
        self.file_path = Path(file_path)
        self.data = {}
        self.last_updated = None
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Rate card file not found: {file_path}")
    
    def parse_impact_rates(self) -> Dict[str, Any]:
        """
        Extract APX Impact pricing from Excel sheets.
        
        Returns:
            Dictionary containing impact rates by market and format
            
        Raises:
            ValueError: If the file format is invalid or data is malformed
        """
        try:
            # Read the APX Impact sheet
            df = pd.read_excel(self.file_path, sheet_name='APX - Impact ')
            
            # The first row contains market codes, starting from column 1
            header_row = df.iloc[0]
            markets = []
            market_columns = []
            
            # Find market columns (skip first column which is format name)
            for i in range(1, len(header_row)):
                market = header_row.iloc[i]
                if pd.notna(market) and isinstance(market, str) and len(str(market).strip()) <= 3:
                    # Market codes are typically 2-3 characters
                    markets.append(str(market).strip())
                    market_columns.append(i)
            
            # Extract format names and rates
            impact_rates = {}
            
            for idx, row in df.iterrows():
                if idx == 0:  # Skip header row
                    continue
                    
                format_name = row.iloc[0]  # Format name is in first column
                
                if pd.isna(format_name) or format_name == '':
                    continue
                    
                # Clean format name
                format_name = str(format_name).strip()
                
                # Extract rates for each market
                format_rates = {}
                for i, market in enumerate(markets):
                    col_idx = market_columns[i]
                    if col_idx < len(row):
                        rate = row.iloc[col_idx]
                        if pd.notna(rate) and str(rate).strip() != '':
                            try:
                                rate_float = float(rate)
                                format_rates[market] = rate_float
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid rate value for {format_name} in {market}: {rate}")
                
                if format_rates:  # Only add if we have valid rates
                    impact_rates[format_name] = format_rates
            
            self.data['impact'] = impact_rates
            logger.info(f"Parsed {len(impact_rates)} Impact formats from rate card")
            return impact_rates
            
        except Exception as e:
            logger.error(f"Error parsing Impact rates: {str(e)}")
            raise ValueError(f"Failed to parse Impact rates: {str(e)}")
    
    def parse_reach_rates(self) -> Dict[str, Any]:
        """
        Extract APX Reach pricing from Excel sheets.
        
        Returns:
            Dictionary containing reach rates by market and format
            
        Raises:
            ValueError: If the file format is invalid or data is malformed
        """
        try:
            # Read the APX Reach sheet
            df = pd.read_excel(self.file_path, sheet_name='APX - Reach')
            
            # Reach sheet has different structure - countries in rows, rate tiers in columns
            # Assuming columns are: Country, Currency, CPM >10K, CPM >20K, CPM >30K, CPM >50K, etc.
            
            reach_rates = {}
            
            # Define standard reach format names based on typical CPM tiers
            format_names = ['CPM >10K', 'CPM >20K', 'CPM >30K', 'CPM >50K']
            
            for idx, row in df.iterrows():
                country = row.iloc[0]  # Country name is in first column
                
                if pd.isna(country) or country == '':
                    continue
                    
                country = str(country).strip()
                
                # Convert country names to market codes (simplified mapping)
                market_mapping = {
                    'Indonesia': 'ID',
                    'Vietnam': 'VN', 
                    'Thailand': 'TH',
                    'Philippines': 'PH',
                    'Malaysia': 'MY',
                    'Singapore': 'SG',
                    'India': 'IN',
                    'New Zealand': 'NZ',
                    'Australia': 'AU',
                    'Hong Kong': 'HK'
                }
                
                market_code = market_mapping.get(country, country[:2].upper())
                
                # Extract rates for different CPM tiers (columns 2-5 typically)
                for i, format_name in enumerate(format_names):
                    col_idx = i + 2  # Start from column 2 (after country and currency)
                    if col_idx < len(row):
                        rate = row.iloc[col_idx]
                        if pd.notna(rate) and str(rate).strip() != '':
                            try:
                                rate_float = float(rate)
                                if format_name not in reach_rates:
                                    reach_rates[format_name] = {}
                                reach_rates[format_name][market_code] = rate_float
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid rate value for {format_name} in {market_code}: {rate}")
            
            self.data['reach'] = reach_rates
            logger.info(f"Parsed {len(reach_rates)} Reach formats from rate card")
            return reach_rates
            
        except Exception as e:
            logger.error(f"Error parsing Reach rates: {str(e)}")
            raise ValueError(f"Failed to parse Reach rates: {str(e)}")
    
    def validate_rate_structure(self) -> bool:
        """
        Validate the integrity and completeness of rate card data.
        
        Returns:
            True if data structure is valid, False otherwise
        """
        try:
            # Check if we have both impact and reach data
            if 'impact' not in self.data or 'reach' not in self.data:
                logger.error("Missing impact or reach data")
                return False
            
            # Check if we have formats and markets
            impact_formats = len(self.data['impact'])
            reach_formats = len(self.data['reach'])
            
            if impact_formats == 0 and reach_formats == 0:
                logger.error("No valid formats found in rate card")
                return False
            
            # Check for consistent market coverage
            all_markets = set()
            for format_data in self.data['impact'].values():
                all_markets.update(format_data.keys())
            for format_data in self.data['reach'].values():
                all_markets.update(format_data.keys())
            
            if len(all_markets) == 0:
                logger.error("No markets found in rate card")
                return False
            
            logger.info(f"Rate card validation passed: {impact_formats} Impact formats, "
                       f"{reach_formats} Reach formats, {len(all_markets)} markets")
            return True
            
        except Exception as e:
            logger.error(f"Rate card validation failed: {str(e)}")
            return False
    
    def get_available_markets(self) -> List[str]:
        """
        Get list of all available markets from the rate card.
        
        Returns:
            List of market codes
        """
        markets = set()
        
        if 'impact' in self.data:
            for format_data in self.data['impact'].values():
                # Only include string market codes, filter out numeric values
                string_markets = [m for m in format_data.keys() if isinstance(m, str)]
                markets.update(string_markets)
        
        if 'reach' in self.data:
            for format_data in self.data['reach'].values():
                # Only include string market codes, filter out numeric values
                string_markets = [m for m in format_data.keys() if isinstance(m, str)]
                markets.update(string_markets)
        
        return sorted(list(markets))
    
    def get_format_rate(self, format_name: str, market: str, product_type: str = 'impact') -> Optional[float]:
        """
        Get rate for a specific format in a specific market.
        
        Args:
            format_name: Name of the ad format
            market: Market code (e.g., 'SG', 'MY')
            product_type: 'impact' or 'reach'
            
        Returns:
            Rate as float, or None if not found
        """
        if product_type not in self.data:
            return None
        
        if format_name not in self.data[product_type]:
            return None
        
        return self.data[product_type][format_name].get(market)
    
    def parse_all_rates(self) -> Dict[str, Any]:
        """
        Parse all rate data from the Excel file.
        
        Returns:
            Complete rate card data structure
        """
        try:
            self.parse_impact_rates()
            self.parse_reach_rates()
            
            if not self.validate_rate_structure():
                raise ValueError("Rate card validation failed")
            
            self.last_updated = datetime.now()
            
            return {
                'impact': self.data.get('impact', {}),
                'reach': self.data.get('reach', {}),
                'markets': self.get_available_markets(),
                'last_updated': self.last_updated
            }
            
        except Exception as e:
            logger.error(f"Failed to parse rate card: {str(e)}")
            raise


class SiteListParser:
    """
    Parser for Excel site list files containing market-specific site categorization data.
    
    Handles site grouping by format and category with validation for data completeness.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the parser with a site list file path.
        
        Args:
            file_path: Path to the Excel site list file
        """
        self.file_path = Path(file_path)
        self.data = {}
        self.last_updated = None
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Site list file not found: {file_path}")
    
    def get_available_markets(self) -> List[str]:
        """
        Get list of available market sheets in the Excel file.
        
        Returns:
            List of market codes
        """
        try:
            excel_file = pd.ExcelFile(self.file_path)
            # Filter out non-market sheets
            excluded_sheets = {'Programmatic', 'Segments'}
            markets = [sheet for sheet in excel_file.sheet_names if sheet not in excluded_sheets]
            return markets
        except Exception as e:
            logger.error(f"Error reading market sheets: {str(e)}")
            return []
    
    def parse_market_sites(self, market: str) -> Dict[str, Any]:
        """
        Extract site data for a specific market.
        
        Args:
            market: Market code (e.g., 'SG', 'MY')
            
        Returns:
            Dictionary containing site data organized by format and category
            
        Raises:
            ValueError: If market sheet is not found or data is malformed
        """
        try:
            df = pd.read_excel(self.file_path, sheet_name=market)
            
            # Initialize market data structure
            market_data = {
                'sites_by_format': {},
                'sites_by_category': {},
                'all_sites': [],
                'formats': set(),
                'categories': set()
            }
            
            # Get format columns (exclude basic info columns)
            basic_columns = {'Publishers', 'Domain', 'Category', 'Remarks'}
            format_columns = [col for col in df.columns if col not in basic_columns and pd.notna(col)]
            
            for _, row in df.iterrows():
                publisher = row.get('Publishers', '')
                domain = row.get('Domain', '')
                category = row.get('Category', '')
                
                if pd.isna(publisher) or publisher == '':
                    continue
                
                # Clean data
                publisher = str(publisher).strip()
                domain = str(domain).strip() if pd.notna(domain) else ''
                category = str(category).strip() if pd.notna(category) else 'Uncategorized'
                
                site_info = {
                    'publisher': publisher,
                    'domain': domain,
                    'category': category,
                    'available_formats': []
                }
                
                # Check which formats are available for this site
                for format_col in format_columns:
                    format_value = row.get(format_col, '')
                    if pd.notna(format_value) and str(format_value).strip().upper() not in ['N.A.', 'N/A', '']:
                        clean_format = format_col.strip()
                        site_info['available_formats'].append(clean_format)
                        market_data['formats'].add(clean_format)
                        
                        # Group by format
                        if clean_format not in market_data['sites_by_format']:
                            market_data['sites_by_format'][clean_format] = []
                        market_data['sites_by_format'][clean_format].append(site_info)
                
                # Group by category
                market_data['categories'].add(category)
                if category not in market_data['sites_by_category']:
                    market_data['sites_by_category'][category] = []
                market_data['sites_by_category'][category].append(site_info)
                
                market_data['all_sites'].append(site_info)
            
            # Convert sets to lists for JSON serialization
            market_data['formats'] = sorted(list(market_data['formats']))
            market_data['categories'] = sorted(list(market_data['categories']))
            
            logger.info(f"Parsed {len(market_data['all_sites'])} sites for market {market}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error parsing sites for market {market}: {str(e)}")
            raise ValueError(f"Failed to parse sites for market {market}: {str(e)}")
    
    def validate_site_data(self, market_data: Dict[str, Any]) -> bool:
        """
        Validate completeness and accuracy of site data.
        
        Args:
            market_data: Parsed market data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check basic structure
            required_keys = ['sites_by_format', 'sites_by_category', 'all_sites', 'formats', 'categories']
            for key in required_keys:
                if key not in market_data:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            # Check if we have sites
            if len(market_data['all_sites']) == 0:
                logger.error("No sites found in market data")
                return False
            
            # Check if we have formats
            if len(market_data['formats']) == 0:
                logger.error("No formats found in market data")
                return False
            
            # Validate site structure
            for site in market_data['all_sites']:
                if not isinstance(site, dict):
                    logger.error("Invalid site data structure")
                    return False
                
                required_site_keys = ['publisher', 'domain', 'category', 'available_formats']
                for key in required_site_keys:
                    if key not in site:
                        logger.error(f"Missing site key: {key}")
                        return False
            
            logger.info(f"Site data validation passed: {len(market_data['all_sites'])} sites, "
                       f"{len(market_data['formats'])} formats, {len(market_data['categories'])} categories")
            return True
            
        except Exception as e:
            logger.error(f"Site data validation failed: {str(e)}")
            return False
    
    def parse_all_markets(self) -> Dict[str, Any]:
        """
        Parse site data for all available markets.
        
        Returns:
            Complete site list data structure
        """
        try:
            markets = self.get_available_markets()
            all_data = {}
            
            for market in markets:
                try:
                    market_data = self.parse_market_sites(market)
                    if self.validate_site_data(market_data):
                        all_data[market] = market_data
                    else:
                        logger.warning(f"Skipping invalid market data for {market}")
                except Exception as e:
                    logger.error(f"Failed to parse market {market}: {str(e)}")
                    continue
            
            if not all_data:
                raise ValueError("No valid market data found")
            
            self.last_updated = datetime.now()
            
            return {
                'markets': all_data,
                'available_markets': list(all_data.keys()),
                'last_updated': self.last_updated
            }
            
        except Exception as e:
            logger.error(f"Failed to parse site lists: {str(e)}")
            raise
    
    def get_sites_by_format(self, market: str, format_name: str) -> List[Dict[str, Any]]:
        """
        Get sites that support a specific format in a market.
        
        Args:
            market: Market code
            format_name: Ad format name
            
        Returns:
            List of site information dictionaries
        """
        if market not in self.data.get('markets', {}):
            return []
        
        market_data = self.data['markets'][market]
        return market_data.get('sites_by_format', {}).get(format_name, [])
    
    def get_sites_by_category(self, market: str, category: str) -> List[Dict[str, Any]]:
        """
        Get sites in a specific category for a market.
        
        Args:
            market: Market code
            category: Site category
            
        Returns:
            List of site information dictionaries
        """
        if market not in self.data.get('markets', {}):
            return []
        
        market_data = self.data['markets'][market]
        return market_data.get('sites_by_category', {}).get(category, [])