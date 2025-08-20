#!/usr/bin/env python3
"""
Demonstration of the data parsing and management system.

This script shows how to use the RateCardParser, SiteListParser, and DataManager
to work with Excel files containing rate cards and site lists.
"""

from data.manager import DataManager
from data.parsers import RateCardParser, SiteListParser


def main():
    """Demonstrate the data parsing and management system."""
    
    print("=== Digital Media Planner Data System Demo ===\n")
    
    # Initialize DataManager
    print("1. Initializing DataManager...")
    manager = DataManager()
    
    # Load and cache data
    print("\n2. Loading rate card data...")
    rate_data = manager.load_rate_cards()
    print(f"   ✓ Loaded {len(rate_data['impact'])} Impact formats")
    print(f"   ✓ Loaded {len(rate_data['reach'])} Reach formats")
    print(f"   ✓ Found {len(rate_data['markets'])} markets in rate card")
    
    print("\n3. Loading site list data...")
    site_data = manager.load_site_lists()
    print(f"   ✓ Loaded {len(site_data['markets'])} markets with site data")
    print(f"   ✓ Available markets: {', '.join(site_data['available_markets'])}")
    
    # Demonstrate market-specific data retrieval
    print("\n4. Getting Singapore (SG) market data...")
    sg_data = manager.get_market_data('SG')
    print(f"   ✓ Market available: {sg_data['available']}")
    print(f"   ✓ Total sites: {sg_data['sites']['total_sites']}")
    print(f"   ✓ Available formats: {', '.join(sg_data['sites']['formats'])}")
    print(f"   ✓ Impact formats with rates: {len(sg_data['rate_card']['impact_formats'])}")
    print(f"   ✓ Reach formats with rates: {len(sg_data['rate_card']['reach_formats'])}")
    
    # Demonstrate specific rate lookup
    print("\n5. Looking up specific rates...")
    desktop_skin_rate = manager.get_format_rate('Desktop Skin', 'SG', 'impact')
    print(f"   ✓ Desktop Skin rate in SG: ${desktop_skin_rate} CPM")
    
    reach_rate = manager.get_format_rate('CPM >10K', 'SG', 'reach')
    print(f"   ✓ Reach CPM >10K rate in SG: ${reach_rate} CPM")
    
    # Demonstrate site lookup by format
    print("\n6. Finding sites that support Desktop Skin in SG...")
    desktop_sites = manager.get_sites_for_format('SG', 'Desktop Skin')
    print(f"   ✓ Found {len(desktop_sites)} sites supporting Desktop Skin")
    if desktop_sites:
        print(f"   ✓ Example site: {desktop_sites[0]['publisher']} ({desktop_sites[0]['domain']})")
    
    # Demonstrate data validation
    print("\n7. Validating data freshness...")
    validation = manager.validate_data_freshness()
    print(f"   ✓ Overall status: {validation['overall_status']}")
    print(f"   ✓ Rate card status: {validation['rate_card']['status']}")
    print(f"   ✓ Site list status: {validation['site_list']['status']}")
    
    # Show cache statistics
    print("\n8. Cache statistics...")
    stats = manager.get_cache_stats()
    print(f"   ✓ Rate card cached in memory: {stats['rate_card']['in_memory']}")
    print(f"   ✓ Site list cached in memory: {stats['site_list']['in_memory']}")
    print(f"   ✓ Cache directory size: {stats['cache_dir_size']:,} bytes")
    
    # Demonstrate all available markets
    print("\n9. All available markets...")
    all_markets = manager.get_available_markets()
    print(f"   ✓ Total markets: {len(all_markets)}")
    print(f"   ✓ Markets: {', '.join(all_markets[:10])}{'...' if len(all_markets) > 10 else ''}")
    
    print("\n=== Demo completed successfully! ===")
    print("\nThe data parsing and management system provides:")
    print("• Robust Excel file parsing with error handling")
    print("• Intelligent caching for performance")
    print("• Data validation and freshness checking")
    print("• Unified API for accessing rate cards and site lists")
    print("• Market-specific data retrieval")
    print("• Format and site lookup capabilities")


if __name__ == "__main__":
    main()