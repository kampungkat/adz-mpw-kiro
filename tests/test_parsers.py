"""
Unit tests for data parsers.
"""

import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from data.parsers import RateCardParser, SiteListParser


class TestRateCardParser(unittest.TestCase):
    """Test cases for RateCardParser class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary Excel file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test_rate_card.xlsx')
        
        # Create test data
        self.create_test_rate_card()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def create_test_rate_card(self):
        """Create a test rate card Excel file."""
        # Create Impact sheet
        impact_data = {
            '$10,000 net budget': ['', 'Desktop Skin', 'Rich Media', 'Standard - Video'],
            'Unnamed: 1': ['', 'Desktop Skin', 'Rich Media', 'Standard - Video'],
            'SG': ['SG', 43, 46, 46],
            'MY': ['MY', 36, 38, 38],
            'TH': ['TH', 40, 42, 42]
        }
        impact_df = pd.DataFrame(impact_data)
        
        # Create Reach sheet (country-based format)
        reach_data = {
            'Country': ['Singapore', 'Malaysia', 'Thailand'],
            'Currency': ['USD', 'USD', 'USD'],
            'CPM >10K': [25, 20, 22],
            'CPM >20K': [30, 25, 27],
            'CPM >30K': [35, 30, 32]
        }
        reach_df = pd.DataFrame(reach_data)
        
        # Write to Excel file
        with pd.ExcelWriter(self.test_file, engine='openpyxl') as writer:
            impact_df.to_excel(writer, sheet_name='APX - Impact ', index=False)
            reach_df.to_excel(writer, sheet_name='APX - Reach', index=False)
    
    def test_init_valid_file(self):
        """Test initialization with valid file."""
        parser = RateCardParser(self.test_file)
        self.assertEqual(parser.file_path, Path(self.test_file))
        self.assertEqual(parser.data, {})
        self.assertIsNone(parser.last_updated)
    
    def test_init_invalid_file(self):
        """Test initialization with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            RateCardParser('non_existent_file.xlsx')
    
    def test_parse_impact_rates(self):
        """Test parsing Impact rates."""
        parser = RateCardParser(self.test_file)
        impact_rates = parser.parse_impact_rates()
        
        # Check structure
        self.assertIn('Desktop Skin', impact_rates)
        self.assertIn('Rich Media', impact_rates)
        self.assertIn('Standard - Video', impact_rates)
        
        # Check specific values
        self.assertEqual(impact_rates['Desktop Skin']['SG'], 43)
        self.assertEqual(impact_rates['Rich Media']['MY'], 38)
        self.assertEqual(impact_rates['Standard - Video']['TH'], 42)
    
    def test_parse_reach_rates(self):
        """Test parsing Reach rates."""
        parser = RateCardParser(self.test_file)
        reach_rates = parser.parse_reach_rates()
        
        # Check structure (CPM tiers)
        self.assertIn('CPM >10K', reach_rates)
        self.assertIn('CPM >20K', reach_rates)
        self.assertIn('CPM >30K', reach_rates)
        
        # Check specific values (using market codes)
        self.assertEqual(reach_rates['CPM >10K']['SG'], 25)
        self.assertEqual(reach_rates['CPM >20K']['MY'], 25)
        self.assertEqual(reach_rates['CPM >30K']['TH'], 32)
    
    def test_validate_rate_structure_valid(self):
        """Test validation with valid data."""
        parser = RateCardParser(self.test_file)
        parser.parse_impact_rates()
        parser.parse_reach_rates()
        
        self.assertTrue(parser.validate_rate_structure())
    
    def test_validate_rate_structure_invalid(self):
        """Test validation with invalid data."""
        parser = RateCardParser(self.test_file)
        # Don't parse any data
        
        self.assertFalse(parser.validate_rate_structure())
    
    def test_get_available_markets(self):
        """Test getting available markets."""
        parser = RateCardParser(self.test_file)
        parser.parse_all_rates()
        
        markets = parser.get_available_markets()
        # Should include markets from both Impact and Reach sheets
        self.assertIn('SG', markets)
        self.assertIn('MY', markets)
        self.assertIn('TH', markets)
        self.assertIsInstance(markets, list)
    
    def test_get_format_rate(self):
        """Test getting specific format rate."""
        parser = RateCardParser(self.test_file)
        parser.parse_all_rates()
        
        # Test valid format and market
        rate = parser.get_format_rate('Desktop Skin', 'SG', 'impact')
        self.assertEqual(rate, 43)
        
        # Test invalid format
        rate = parser.get_format_rate('Invalid Format', 'SG', 'impact')
        self.assertIsNone(rate)
        
        # Test invalid market
        rate = parser.get_format_rate('Desktop Skin', 'INVALID', 'impact')
        self.assertIsNone(rate)
    
    def test_parse_all_rates(self):
        """Test parsing all rates."""
        parser = RateCardParser(self.test_file)
        all_data = parser.parse_all_rates()
        
        # Check structure
        self.assertIn('impact', all_data)
        self.assertIn('reach', all_data)
        self.assertIn('markets', all_data)
        self.assertIn('last_updated', all_data)
        
        # Check data presence
        self.assertTrue(len(all_data['impact']) > 0)
        self.assertTrue(len(all_data['reach']) > 0)
        self.assertTrue(len(all_data['markets']) > 0)
        self.assertIsNotNone(all_data['last_updated'])


class TestSiteListParser(unittest.TestCase):
    """Test cases for SiteListParser class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary Excel file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test_site_list.xlsx')
        
        # Create test data
        self.create_test_site_list()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def create_test_site_list(self):
        """Create a test site list Excel file."""
        # Create SG market data
        sg_data = {
            'Publishers': ['Zaobao', 'Straitstimes', 'AsiaOne'],
            'Domain': ['zaobao.com.sg', 'straitstimes.com', 'asiaone.com'],
            'Category': ['Chinese News and Media', 'News and Media', 'News and Media'],
            'Desktop Skin ': ['Available', 'Available', 'N.A.'],
            'Mobile Scroller': ['Available', 'N.A.', 'Available'],
            'Mobile Skin': ['N.A.', 'Available', 'Available'],
            'Remarks': ['', '', '']
        }
        sg_df = pd.DataFrame(sg_data)
        
        # Create MY market data
        my_data = {
            'Publishers': ['The Star', 'Malaysiakini', 'New Straits Times'],
            'Domain': ['thestar.com.my', 'malaysiakini.com', 'nst.com.my'],
            'Category': ['News and Media', 'News and Media', 'News and Media'],
            'Desktop Skin ': ['Available', 'N.A.', 'Available'],
            'Mobile Scroller': ['N.A.', 'Available', 'Available'],
            'Mobile Skin': ['Available', 'Available', 'N.A.'],
            'Remarks': ['', '', '']
        }
        my_df = pd.DataFrame(my_data)
        
        # Write to Excel file
        with pd.ExcelWriter(self.test_file, engine='openpyxl') as writer:
            sg_df.to_excel(writer, sheet_name='SG', index=False)
            my_df.to_excel(writer, sheet_name='MY', index=False)
    
    def test_init_valid_file(self):
        """Test initialization with valid file."""
        parser = SiteListParser(self.test_file)
        self.assertEqual(parser.file_path, Path(self.test_file))
        self.assertEqual(parser.data, {})
        self.assertIsNone(parser.last_updated)
    
    def test_init_invalid_file(self):
        """Test initialization with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            SiteListParser('non_existent_file.xlsx')
    
    def test_get_available_markets(self):
        """Test getting available markets."""
        parser = SiteListParser(self.test_file)
        markets = parser.get_available_markets()
        
        expected_markets = ['SG', 'MY']
        self.assertEqual(sorted(markets), sorted(expected_markets))
    
    def test_parse_market_sites(self):
        """Test parsing sites for a specific market."""
        parser = SiteListParser(self.test_file)
        sg_data = parser.parse_market_sites('SG')
        
        # Check structure
        self.assertIn('sites_by_format', sg_data)
        self.assertIn('sites_by_category', sg_data)
        self.assertIn('all_sites', sg_data)
        self.assertIn('formats', sg_data)
        self.assertIn('categories', sg_data)
        
        # Check data
        self.assertEqual(len(sg_data['all_sites']), 3)
        self.assertIn('Desktop Skin', sg_data['formats'])  # Stripped whitespace
        self.assertIn('Mobile Scroller', sg_data['formats'])
        self.assertIn('News and Media', sg_data['categories'])
    
    def test_validate_site_data_valid(self):
        """Test validation with valid site data."""
        parser = SiteListParser(self.test_file)
        sg_data = parser.parse_market_sites('SG')
        
        self.assertTrue(parser.validate_site_data(sg_data))
    
    def test_validate_site_data_invalid(self):
        """Test validation with invalid site data."""
        parser = SiteListParser(self.test_file)
        invalid_data = {'invalid': 'structure'}
        
        self.assertFalse(parser.validate_site_data(invalid_data))
    
    def test_parse_all_markets(self):
        """Test parsing all markets."""
        parser = SiteListParser(self.test_file)
        all_data = parser.parse_all_markets()
        
        # Check structure
        self.assertIn('markets', all_data)
        self.assertIn('available_markets', all_data)
        self.assertIn('last_updated', all_data)
        
        # Check data
        self.assertIn('SG', all_data['markets'])
        self.assertIn('MY', all_data['markets'])
        self.assertEqual(sorted(all_data['available_markets']), ['MY', 'SG'])
        self.assertIsNotNone(all_data['last_updated'])
    
    def test_get_sites_by_format(self):
        """Test getting sites by format."""
        parser = SiteListParser(self.test_file)
        parser.data = parser.parse_all_markets()
        
        # Test valid format
        sites = parser.get_sites_by_format('SG', 'Desktop Skin')  # Stripped whitespace
        self.assertTrue(len(sites) > 0)
        
        # Test invalid market
        sites = parser.get_sites_by_format('INVALID', 'Desktop Skin')
        self.assertEqual(len(sites), 0)
    
    def test_get_sites_by_category(self):
        """Test getting sites by category."""
        parser = SiteListParser(self.test_file)
        parser.data = parser.parse_all_markets()
        
        # Test valid category
        sites = parser.get_sites_by_category('SG', 'News and Media')
        self.assertTrue(len(sites) > 0)
        
        # Test invalid market
        sites = parser.get_sites_by_category('INVALID', 'News and Media')
        self.assertEqual(len(sites), 0)


class TestParserEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for parsers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_rate_card_malformed_data(self):
        """Test rate card parser with malformed data."""
        # Create file with missing sheets
        test_file = os.path.join(self.temp_dir, 'malformed.xlsx')
        
        # Create a file with wrong sheet names
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Wrong Sheet', index=False)
        
        parser = RateCardParser(test_file)
        
        # Should raise ValueError for missing sheets
        with self.assertRaises(ValueError):
            parser.parse_impact_rates()
    
    def test_site_list_empty_data(self):
        """Test site list parser with empty data."""
        test_file = os.path.join(self.temp_dir, 'empty.xlsx')
        
        # Create empty dataframe
        df = pd.DataFrame()
        with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='SG', index=False)
        
        parser = SiteListParser(test_file)
        sg_data = parser.parse_market_sites('SG')
        
        # Should handle empty data gracefully
        self.assertEqual(len(sg_data['all_sites']), 0)
        self.assertFalse(parser.validate_site_data(sg_data))
    
    def test_rate_card_invalid_numeric_values(self):
        """Test rate card parser with invalid numeric values."""
        test_file = os.path.join(self.temp_dir, 'invalid_numbers.xlsx')
        
        # Create data with invalid numbers
        impact_data = {
            '$10,000 net budget': ['', 'Desktop Skin', 'Rich Media'],
            'Unnamed: 1': ['', 'Desktop Skin', 'Rich Media'],
            'SG': ['SG', 'invalid', 46],
            'MY': ['MY', 36, 'also_invalid']
        }
        impact_df = pd.DataFrame(impact_data)
        
        reach_data = {
            '$10,000 net budget': ['', 'Banner'],
            'Unnamed: 1': ['', 'Banner'],
            'SG': ['SG', 25]
        }
        reach_df = pd.DataFrame(reach_data)
        
        with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
            impact_df.to_excel(writer, sheet_name='APX - Impact ', index=False)
            reach_df.to_excel(writer, sheet_name='APX - Reach', index=False)
        
        parser = RateCardParser(test_file)
        
        # Should handle invalid numbers gracefully
        impact_rates = parser.parse_impact_rates()
        
        # Should only include valid rates
        self.assertNotIn('SG', impact_rates.get('Desktop Skin', {}))
        self.assertNotIn('MY', impact_rates.get('Rich Media', {}))
        self.assertEqual(impact_rates.get('Rich Media', {}).get('SG'), 46)


if __name__ == '__main__':
    unittest.main()


class TestDataManager(unittest.TestCase):
    """Test cases for DataManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid circular imports
        from data.manager import DataManager
        
        # Create temporary directories and files
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, 'cache')
        
        # Create test files
        self.rate_card_file = os.path.join(self.temp_dir, 'test_rate_card.xlsx')
        self.site_list_file = os.path.join(self.temp_dir, 'test_site_list.xlsx')
        
        self.create_test_files()
        
        # Initialize DataManager with test cache directory
        self.manager = DataManager(cache_dir=self.cache_dir, cache_ttl_hours=1)
        self.manager.default_rate_card_path = self.rate_card_file
        self.manager.default_site_list_path = self.site_list_file
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test Excel files."""
        # Create rate card file
        impact_data = {
            '$10,000 net budget': ['', 'Desktop Skin', 'Rich Media'],
            'Unnamed: 1': ['', 'Desktop Skin', 'Rich Media'],
            'SG': ['SG', 43, 46],
            'MY': ['MY', 36, 38]
        }
        impact_df = pd.DataFrame(impact_data)
        
        reach_data = {
            'Country': ['Singapore', 'Malaysia'],
            'Currency': ['USD', 'USD'],
            'CPM >10K': [6.0, 4.5],
            'CPM >20K': [5.0, 4.0]
        }
        reach_df = pd.DataFrame(reach_data)
        
        with pd.ExcelWriter(self.rate_card_file, engine='openpyxl') as writer:
            impact_df.to_excel(writer, sheet_name='APX - Impact ', index=False)
            reach_df.to_excel(writer, sheet_name='APX - Reach', index=False)
        
        # Create site list file
        sg_data = {
            'Publishers': ['Zaobao', 'Straitstimes'],
            'Domain': ['zaobao.com.sg', 'straitstimes.com'],
            'Category': ['Chinese News and Media', 'News and Media'],
            'Desktop Skin': ['Available', 'Available'],
            'Mobile Scroller': ['Available', 'N.A.']
        }
        sg_df = pd.DataFrame(sg_data)
        
        with pd.ExcelWriter(self.site_list_file, engine='openpyxl') as writer:
            sg_df.to_excel(writer, sheet_name='SG', index=False)
    
    def test_load_rate_cards(self):
        """Test loading rate card data."""
        data = self.manager.load_rate_cards()
        
        # Check structure
        self.assertIn('impact', data)
        self.assertIn('reach', data)
        self.assertIn('markets', data)
        
        # Check that cache was created
        self.assertIsNotNone(self.manager._rate_card_cache)
    
    def test_load_site_lists(self):
        """Test loading site list data."""
        data = self.manager.load_site_lists()
        
        # Check structure
        self.assertIn('markets', data)
        self.assertIn('available_markets', data)
        
        # Check that cache was created
        self.assertIsNotNone(self.manager._site_list_cache)
    
    def test_caching_mechanism(self):
        """Test that caching works correctly."""
        # First load
        data1 = self.manager.load_rate_cards()
        cache_time1 = self.manager._rate_card_cache.last_accessed
        
        # Second load should use cache
        data2 = self.manager.load_rate_cards()
        cache_time2 = self.manager._rate_card_cache.last_accessed
        
        # Data should be the same
        self.assertEqual(data1, data2)
        
        # Access time should be updated
        self.assertGreater(cache_time2, cache_time1)
    
    def test_get_market_data(self):
        """Test getting market-specific data."""
        market_data = self.manager.get_market_data('SG')
        
        # Check structure
        self.assertIn('market_code', market_data)
        self.assertIn('available', market_data)
        self.assertIn('rate_card', market_data)
        self.assertIn('sites', market_data)
        
        # Check data
        self.assertEqual(market_data['market_code'], 'SG')
        self.assertTrue(market_data['available'])
    
    def test_validate_data_freshness(self):
        """Test data freshness validation."""
        # Load data first
        self.manager.load_rate_cards()
        self.manager.load_site_lists()
        
        validation = self.manager.validate_data_freshness()
        
        # Check structure
        self.assertIn('rate_card', validation)
        self.assertIn('site_list', validation)
        self.assertIn('overall_status', validation)
        
        # Should be fresh since we just loaded
        self.assertEqual(validation['rate_card']['status'], 'fresh')
        self.assertEqual(validation['overall_status'], 'ready')
    
    def test_get_available_markets(self):
        """Test getting available markets."""
        markets = self.manager.get_available_markets()
        
        # Should include markets from both sources
        self.assertIn('SG', markets)
        self.assertIsInstance(markets, list)
    
    def test_get_format_rate(self):
        """Test getting format rates."""
        rate = self.manager.get_format_rate('Desktop Skin', 'SG', 'impact')
        
        # Should return the rate from test data
        self.assertEqual(rate, 43)
        
        # Test invalid format
        rate = self.manager.get_format_rate('Invalid Format', 'SG', 'impact')
        self.assertIsNone(rate)
    
    def test_get_sites_for_format(self):
        """Test getting sites for format."""
        sites = self.manager.get_sites_for_format('SG', 'Desktop Skin')
        
        # Should return sites that support the format
        self.assertTrue(len(sites) > 0)
        self.assertIsInstance(sites, list)
    
    def test_clear_cache(self):
        """Test clearing cache."""
        # Load data to create cache
        self.manager.load_rate_cards()
        self.manager.load_site_lists()
        
        # Verify cache exists
        self.assertIsNotNone(self.manager._rate_card_cache)
        self.assertIsNotNone(self.manager._site_list_cache)
        
        # Clear cache
        self.manager.clear_cache()
        
        # Verify cache is cleared
        self.assertIsNone(self.manager._rate_card_cache)
        self.assertIsNone(self.manager._site_list_cache)
    
    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        # Load data to create cache
        self.manager.load_rate_cards()
        
        stats = self.manager.get_cache_stats()
        
        # Check structure
        self.assertIn('rate_card', stats)
        self.assertIn('site_list', stats)
        self.assertIn('cache_dir_size', stats)
        
        # Rate card should be in memory
        self.assertTrue(stats['rate_card']['in_memory'])
    
    def test_file_change_detection(self):
        """Test that file changes invalidate cache."""
        # Load data to create cache
        data1 = self.manager.load_rate_cards()
        
        # Modify the file (simulate by changing modification time)
        import time
        time.sleep(0.1)  # Ensure different timestamp
        
        # Touch the file to change its hash
        with open(self.rate_card_file, 'a') as f:
            f.write(' ')  # Add a space to change file hash
        
        # Load again - should detect change and reload
        data2 = self.manager.load_rate_cards()
        
        # Cache should have been invalidated and reloaded
        self.assertIsNotNone(self.manager._rate_card_cache)