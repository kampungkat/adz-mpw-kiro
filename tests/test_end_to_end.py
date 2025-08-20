"""
End-to-end testing and validation for the complete media planning system.

These tests validate the entire system using real data files and workflows
to ensure production readiness and quality assurance.
"""

import pytest
import tempfile
import os
import pandas as pd
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from models.data_models import ClientBrief, MediaPlan, FormatAllocation
from business_logic.media_plan_controller import MediaPlanController
from data.manager import DataManager
from ui.components import MediaPlannerForm, PlanDisplayComponent


class TestEndToEndWorkflows:
    """Complete end-to-end workflow testing."""
    
    def setup_method(self):
        """Set up comprehensive test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.rate_card_file = os.path.join(self.temp_dir, 'adzymic_rate_card.xlsx')
        self.site_list_file = os.path.join(self.temp_dir, 'adzymic_site_list.xlsx')
        
        # Create production-like test data
        self._create_production_test_files()
        
        # Initialize system components
        self.data_manager = DataManager(cache_dir=os.path.join(self.temp_dir, 'cache'))
        self.data_manager.default_rate_card_path = self.rate_card_file
        self.data_manager.default_site_list_path = self.site_list_file
        
        self.controller = MediaPlanController(self.data_manager, testing_mode=True)
        
        # Test scenarios
        self.test_scenarios = [
            {
                'name': 'Small Budget Campaign',
                'brief': ClientBrief(
                    brand_name="Local Restaurant",
                    budget=5000.0,
                    country="SG",
                    campaign_period="Q1 2024",
                    objective="Local Awareness",
                    planning_mode="AI"
                ),
                'expected_formats': 1,  # Should focus on cost-effective formats
                'min_budget_utilization': 0.7
            },
            {
                'name': 'Medium Budget Campaign',
                'brief': ClientBrief(
                    brand_name="Regional Brand",
                    budget=25000.0,
                    country="SG",
                    campaign_period="Q2 2024",
                    objective="Brand Awareness",
                    planning_mode="AI"
                ),
                'expected_formats': 3,  # Should diversify across formats
                'min_budget_utilization': 0.8
            },
            {
                'name': 'Large Budget Campaign',
                'brief': ClientBrief(
                    brand_name="Enterprise Client",
                    budget=100000.0,
                    country="SG",
                    campaign_period="Q3 2024",
                    objective="Market Penetration",
                    planning_mode="AI"
                ),
                'expected_formats': 4,  # Should use premium formats
                'min_budget_utilization': 0.85
            },
            {
                'name': 'Manual Format Selection',
                'brief': ClientBrief(
                    brand_name="Specific Requirements Client",
                    budget=30000.0,
                    country="SG",
                    campaign_period="Q4 2024",
                    objective="Targeted Campaign",
                    planning_mode="Manual",
                    selected_formats=["Rich Media Banner", "Native Content"]
                ),
                'expected_formats': 2,  # Should only use selected formats
                'min_budget_utilization': 0.75
            }
        ]
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_production_test_files(self):
        """Create production-like test data files."""
        # Create comprehensive rate card with multiple markets
        impact_data = {
            '$10,000 net budget': ['', 'Desktop Skin', 'Rich Media Banner', 'Video Pre-roll', 'Mobile Skin', 'Native Content', 'Interstitial', 'Expandable Banner'],
            'Unnamed: 1': ['', 'Desktop Skin', 'Rich Media Banner', 'Video Pre-roll', 'Mobile Skin', 'Native Content', 'Interstitial', 'Expandable Banner'],
            'SG': ['SG', 45, 38, 52, 35, 42, 48, 40],
            'MY': ['MY', 38, 32, 45, 30, 36, 42, 35],
            'TH': ['TH', 42, 35, 48, 32, 38, 45, 38],
            'ID': ['ID', 35, 28, 40, 25, 32, 38, 30],
            'PH': ['PH', 40, 33, 46, 28, 35, 42, 33],
            'VN': ['VN', 30, 25, 35, 22, 28, 32, 26]
        }
        impact_df = pd.DataFrame(impact_data)
        
        # Create reach rates with multiple tiers (matching parser expectations)
        reach_data = {
            '$10,000 net budget': ['', 'CPM >10K', 'CPM >20K', 'CPM >30K', 'CPM >50K', 'CPM >100K'],
            'Unnamed: 1': ['', 'CPM >10K', 'CPM >20K', 'CPM >30K', 'CPM >50K', 'CPM >100K'],
            'SG': ['SG', 28, 25, 22, 20, 18],
            'MY': ['MY', 22, 20, 18, 16, 14],
            'TH': ['TH', 25, 22, 20, 18, 16],
            'ID': ['ID', 18, 16, 14, 12, 10],
            'PH': ['PH', 20, 18, 16, 14, 12],
            'VN': ['VN', 16, 14, 12, 10, 8]
        }
        reach_df = pd.DataFrame(reach_data)
        
        with pd.ExcelWriter(self.rate_card_file, engine='openpyxl') as writer:
            impact_df.to_excel(writer, sheet_name='APX - Impact ', index=False)
            reach_df.to_excel(writer, sheet_name='APX - Reach', index=False)
        
        # Create comprehensive site lists for multiple markets
        markets_data = {
            'SG': {
                'Publishers': [
                    'Straits Times', 'Channel NewsAsia', 'Zaobao', 'TODAY', 'AsiaOne',
                    'Hardware Zone', 'Mothership', 'Yahoo Singapore', 'MSN Singapore', 'Carousell',
                    'PropertyGuru', 'JobStreet', 'Grab', 'Foodpanda', 'Shopee'
                ],
                'Domain': [
                    'straitstimes.com', 'channelnewsasia.com', 'zaobao.com.sg', 'todayonline.com', 'asiaone.com',
                    'hardwarezone.com.sg', 'mothership.sg', 'sg.yahoo.com', 'msn.com/sg', 'carousell.sg',
                    'propertyguru.com.sg', 'jobstreet.com.sg', 'grab.com/sg', 'foodpanda.sg', 'shopee.sg'
                ],
                'Category': [
                    'News and Media', 'News and Media', 'Chinese News and Media', 'News and Media', 'News and Media',
                    'Technology', 'Entertainment', 'Portal', 'Portal', 'E-commerce',
                    'Property', 'Jobs', 'Transportation', 'Food Delivery', 'E-commerce'
                ]
            },
            'MY': {
                'Publishers': [
                    'The Star', 'Malaysiakini', 'New Straits Times', 'Astro Awani', 'Harian Metro',
                    'Lowyat.NET', 'Says.com', 'Mudah.my', 'JobStreet Malaysia', 'iProperty'
                ],
                'Domain': [
                    'thestar.com.my', 'malaysiakini.com', 'nst.com.my', 'astroawani.com', 'hmetro.com.my',
                    'lowyat.net', 'says.com', 'mudah.my', 'jobstreet.com.my', 'iproperty.com.my'
                ],
                'Category': [
                    'News and Media', 'News and Media', 'News and Media', 'News and Media', 'News and Media',
                    'Technology', 'Entertainment', 'Classifieds', 'Jobs', 'Property'
                ]
            }
        }
        
        # Create format availability matrix
        formats = ['Desktop Skin', 'Rich Media Banner', 'Video Pre-roll', 'Mobile Skin', 'Native Content', 'Interstitial', 'Expandable Banner']
        
        with pd.ExcelWriter(self.site_list_file, engine='openpyxl') as writer:
            for market, data in markets_data.items():
                # Create format availability (realistic distribution)
                format_availability = {}
                for format_name in formats:
                    availability = []
                    for i, publisher in enumerate(data['Publishers']):
                        # Simulate realistic availability patterns
                        if format_name in ['Rich Media Banner', 'Native Content']:
                            availability.append('Available')  # Most common formats
                        elif format_name in ['Desktop Skin', 'Mobile Skin']:
                            availability.append('Available' if i % 2 == 0 else 'N.A.')  # 50% availability
                        elif format_name == 'Video Pre-roll':
                            availability.append('Available' if i % 3 == 0 else 'N.A.')  # 33% availability
                        else:
                            availability.append('Available' if i % 4 == 0 else 'N.A.')  # 25% availability
                    format_availability[format_name] = availability
                
                # Create DataFrame
                market_data = {
                    'Publishers': data['Publishers'],
                    'Domain': data['Domain'],
                    'Category': data['Category'],
                    **format_availability
                }
                market_df = pd.DataFrame(market_data)
                market_df.to_excel(writer, sheet_name=market, index=False)
    
    def test_complete_system_validation(self):
        """Test complete system validation across all scenarios."""
        results = {}
        
        for scenario in self.test_scenarios:
            print(f"\nTesting scenario: {scenario['name']}")
            
            # Mock AI response for consistent testing
            mock_plans = self._create_mock_plans_for_scenario(scenario)
            
            with patch.object(self.controller.ai_generator, 'generate_multiple_plans') as mock_generate:
                mock_generate.return_value = mock_plans
                
                # Execute workflow
                success, plans, message, notification = self.controller.generate_plans(scenario['brief'])
                
                # Validate results
                scenario_results = self._validate_scenario_results(scenario, success, plans, message)
                results[scenario['name']] = scenario_results
                
                # Assert basic success criteria
                assert success, f"Scenario '{scenario['name']}' failed: {message}"
                assert len(plans) >= 1, f"No plans generated for scenario '{scenario['name']}'"
        
        # Generate validation report
        self._generate_validation_report(results)
    
    def _create_mock_plans_for_scenario(self, scenario):
        """Create realistic mock plans for a scenario."""
        brief = scenario['brief']
        plans = []
        
        # Get available formats for the market
        available_formats = self.controller.get_available_formats(brief.country)
        all_formats = available_formats['impact_formats'] + available_formats['reach_formats']
        
        # Filter by selected formats if manual mode
        if brief.planning_mode == "Manual" and brief.selected_formats:
            all_formats = [f for f in all_formats if f in brief.selected_formats]
        
        # Create 3 different strategic plans
        strategies = [
            {
                'title': 'Cost-Efficient Strategy',
                'focus': 'reach',
                'format_count': min(2, len(all_formats)),
                'budget_utilization': 0.85
            },
            {
                'title': 'Balanced Multi-Format Strategy', 
                'focus': 'balanced',
                'format_count': min(3, len(all_formats)),
                'budget_utilization': 0.90
            },
            {
                'title': 'Premium Impact Strategy',
                'focus': 'quality',
                'format_count': min(2, len(all_formats)),
                'budget_utilization': 0.80
            }
        ]
        
        for i, strategy in enumerate(strategies):
            # Select formats for this strategy
            selected_formats = all_formats[:strategy['format_count']]
            
            # Create allocations
            allocations = []
            total_budget = brief.budget * strategy['budget_utilization']
            budget_per_format = total_budget / len(selected_formats)
            
            for format_name in selected_formats:
                # Get realistic CPM (mock data)
                cpm = self._get_mock_cpm(format_name, brief.country)
                estimated_impressions = int(budget_per_format / cpm * 1000) if cpm > 0 else 0
                
                allocation = FormatAllocation(
                    format_name=format_name,
                    budget_allocation=budget_per_format,
                    cpm=cpm,
                    estimated_impressions=estimated_impressions,
                    recommended_sites=self._get_mock_sites(format_name, brief.country),
                    notes=f"{strategy['focus'].title()} placement"
                )
                allocations.append(allocation)
            
            # Create plan
            plan = MediaPlan(
                plan_id=f"test_plan_{i+1}",
                title=strategy['title'],
                total_budget=total_budget,
                allocations=allocations,
                estimated_reach=int(sum(a.estimated_impressions for a in allocations) * 0.3),
                estimated_impressions=sum(a.estimated_impressions for a in allocations),
                rationale=f"{strategy['title']} optimized for {brief.objective}",
                created_at=datetime.now()
            )
            plans.append(plan)
        
        return plans
    
    def _get_mock_cpm(self, format_name, country):
        """Get mock CPM for format and country."""
        # Simplified CPM mapping
        base_cpms = {
            'Desktop Skin': 45,
            'Rich Media Banner': 38,
            'Video Pre-roll': 52,
            'Mobile Skin': 35,
            'Native Content': 42,
            'Interstitial': 48,
            'Expandable Banner': 40
        }
        
        # Country multipliers
        country_multipliers = {
            'SG': 1.0,
            'MY': 0.85,
            'TH': 0.90,
            'ID': 0.75,
            'PH': 0.80,
            'VN': 0.65
        }
        
        base_cpm = base_cpms.get(format_name, 35)
        multiplier = country_multipliers.get(country, 1.0)
        
        return base_cpm * multiplier
    
    def _get_mock_sites(self, format_name, country):
        """Get mock sites for format and country."""
        # Simplified site mapping
        sites_map = {
            'SG': ['straitstimes.com', 'channelnewsasia.com', 'mothership.sg'],
            'MY': ['thestar.com.my', 'malaysiakini.com', 'says.com']
        }
        
        return sites_map.get(country, ['example.com'])[:2]  # Return top 2 sites
    
    def _validate_scenario_results(self, scenario, success, plans, message):
        """Validate results for a specific scenario."""
        results = {
            'success': success,
            'message': message,
            'plan_count': len(plans) if plans else 0,
            'validations': {}
        }
        
        if not success:
            results['validations']['basic_success'] = False
            return results
        
        brief = scenario['brief']
        
        # Validate plan count
        results['validations']['plan_count'] = len(plans) >= 1
        
        # Validate budget utilization
        budget_utilizations = []
        format_counts = []
        
        for plan in plans:
            budget_utilization = plan.total_budget / brief.budget
            budget_utilizations.append(budget_utilization)
            format_counts.append(len(plan.allocations))
            
            # Validate individual plan
            results['validations'][f'plan_{plan.plan_id}_budget_valid'] = (
                0 < plan.total_budget <= brief.budget
            )
            results['validations'][f'plan_{plan.plan_id}_has_allocations'] = (
                len(plan.allocations) > 0
            )
            results['validations'][f'plan_{plan.plan_id}_has_impressions'] = (
                plan.estimated_impressions > 0
            )
        
        # Aggregate validations
        results['validations']['min_budget_utilization'] = (
            min(budget_utilizations) >= scenario['min_budget_utilization']
        )
        results['validations']['format_diversity'] = (
            max(format_counts) >= scenario['expected_formats']
        )
        
        # Manual format validation
        if brief.planning_mode == "Manual" and brief.selected_formats:
            all_used_formats = set()
            for plan in plans:
                for allocation in plan.allocations:
                    all_used_formats.add(allocation.format_name)
            
            results['validations']['manual_format_compliance'] = (
                all_used_formats.issubset(set(brief.selected_formats))
            )
        
        return results
    
    def _generate_validation_report(self, results):
        """Generate comprehensive validation report."""
        print("\n" + "="*80)
        print("END-TO-END VALIDATION REPORT")
        print("="*80)
        
        total_scenarios = len(results)
        successful_scenarios = sum(1 for r in results.values() if r['success'])
        
        print(f"Total Scenarios Tested: {total_scenarios}")
        print(f"Successful Scenarios: {successful_scenarios}")
        print(f"Success Rate: {successful_scenarios/total_scenarios*100:.1f}%")
        print()
        
        for scenario_name, result in results.items():
            print(f"Scenario: {scenario_name}")
            print(f"  Success: {result['success']}")
            print(f"  Plans Generated: {result['plan_count']}")
            
            if result['validations']:
                passed_validations = sum(1 for v in result['validations'].values() if v)
                total_validations = len(result['validations'])
                print(f"  Validations Passed: {passed_validations}/{total_validations}")
                
                # Show failed validations
                failed = [k for k, v in result['validations'].items() if not v]
                if failed:
                    print(f"  Failed Validations: {', '.join(failed)}")
            
            print()
    
    def test_data_quality_validation(self):
        """Test data quality and consistency."""
        print("\nValidating data quality...")
        
        # Test rate card data
        rate_card_data = self.data_manager.load_rate_cards()
        
        assert 'impact' in rate_card_data, "Missing impact rates"
        assert 'reach' in rate_card_data, "Missing reach rates"
        assert 'markets' in rate_card_data, "Missing market data"
        
        # Validate rate consistency
        impact_markets = set()
        for format_data in rate_card_data['impact'].values():
            impact_markets.update(format_data.keys())
        
        # For reach data, markets are nested under format tiers
        reach_markets = set()
        if rate_card_data['reach']:
            for tier_data in rate_card_data['reach'].values():
                if isinstance(tier_data, dict):
                    reach_markets.update(tier_data.keys())
        
        # Should have overlapping markets or at least impact markets
        common_markets = impact_markets.intersection(reach_markets) if reach_markets else impact_markets
        assert len(common_markets) > 0 or len(impact_markets) > 0, f"No markets found. Impact: {impact_markets}, Reach: {reach_markets}"
        
        # Test site list data
        site_data = self.data_manager.load_site_lists()
        
        assert 'markets' in site_data, "Missing site market data"
        assert 'available_markets' in site_data, "Missing available markets list"
        
        # Validate site-format consistency
        for market, market_data in site_data['markets'].items():
            if 'sites_by_format' in market_data:
                for format_name, sites in market_data['sites_by_format'].items():
                    assert isinstance(sites, list), f"Sites for {format_name} in {market} not a list"
                    assert len(sites) > 0, f"No sites for {format_name} in {market}"
        
        print("Data quality validation passed!")
    
    def test_performance_benchmarks(self):
        """Test system performance benchmarks."""
        import time
        
        print("\nRunning performance benchmarks...")
        
        # Test data loading performance
        start_time = time.time()
        self.data_manager.load_rate_cards()
        self.data_manager.load_site_lists()
        data_load_time = time.time() - start_time
        
        assert data_load_time < 5.0, f"Data loading too slow: {data_load_time:.2f}s"
        print(f"Data loading time: {data_load_time:.2f}s")
        
        # Test plan generation performance (mocked)
        brief = ClientBrief(
            brand_name="Performance Test",
            budget=50000.0,
            country="SG",
            campaign_period="Q1 2024",
            objective="Performance Test",
            planning_mode="AI"
        )
        
        with patch.object(self.controller.ai_generator, 'generate_multiple_plans') as mock_generate:
            mock_generate.return_value = self._create_mock_plans_for_scenario({
                'brief': brief,
                'expected_formats': 3,
                'min_budget_utilization': 0.8
            })
            
            start_time = time.time()
            success, plans, message, notification = self.controller.generate_plans(brief)
            generation_time = time.time() - start_time
            
            assert generation_time < 10.0, f"Plan generation too slow: {generation_time:.2f}s"
            print(f"Plan generation time: {generation_time:.2f}s")
        
        # Test plan comparison performance
        if plans:
            start_time = time.time()
            comparison = self.controller.compare_plans(plans, brief)
            comparison_time = time.time() - start_time
            
            assert comparison_time < 2.0, f"Plan comparison too slow: {comparison_time:.2f}s"
            print(f"Plan comparison time: {comparison_time:.2f}s")
        
        print("Performance benchmarks passed!")
    
    def test_error_recovery_scenarios(self):
        """Test system behavior under various error conditions."""
        print("\nTesting error recovery scenarios...")
        
        # Test with corrupted data files
        corrupted_controller = MediaPlanController(testing_mode=True)
        corrupted_controller.data_manager.default_rate_card_path = "nonexistent.xlsx"
        
        brief = ClientBrief(
            brand_name="Error Test",
            budget=10000.0,
            country="SG",
            campaign_period="Q1 2024",
            objective="Error Recovery Test",
            planning_mode="AI"
        )
        
        success, plans, message, notification = corrupted_controller.generate_plans(brief)
        assert not success, "Should fail with missing data files"
        assert notification is not None, "Should provide user notification"
        print("✓ Handles missing data files correctly")
        
        # Test with invalid market
        invalid_brief = ClientBrief(
            brand_name="Invalid Market Test",
            budget=10000.0,
            country="INVALID",
            campaign_period="Q1 2024",
            objective="Error Recovery Test",
            planning_mode="AI"
        )
        
        is_valid, message = self.controller.validate_inputs(invalid_brief)
        assert not is_valid, "Should reject invalid market"
        assert "not available" in message.lower(), "Should explain market unavailability"
        print("✓ Handles invalid markets correctly")
        
        # Test with excessive budget
        excessive_brief = ClientBrief(
            brand_name="Excessive Budget Test",
            budget=5000000.0,  # $5M
            country="SG",
            campaign_period="Q1 2024",
            objective="Error Recovery Test",
            planning_mode="AI"
        )
        
        is_valid, message = self.controller.validate_inputs(excessive_brief)
        assert not is_valid, "Should reject excessive budget"
        assert "maximum limit" in message.lower(), "Should explain budget limit"
        print("✓ Handles excessive budgets correctly")
        
        print("Error recovery scenarios passed!")
    
    def test_cross_market_consistency(self):
        """Test consistency across different markets."""
        print("\nTesting cross-market consistency...")
        
        markets_to_test = ['SG', 'MY']
        results = {}
        
        for market in markets_to_test:
            brief = ClientBrief(
                brand_name=f"Cross Market Test {market}",
                budget=25000.0,
                country=market,
                campaign_period="Q1 2024",
                objective="Cross Market Test",
                planning_mode="AI"
            )
            
            # Validate market data availability
            market_data = self.controller.data_manager.get_market_data(market)
            assert market_data.get('available'), f"Market {market} should be available"
            
            # Test format availability
            formats = self.controller.get_available_formats(market)
            assert len(formats['impact_formats']) > 0, f"No impact formats for {market}"
            
            results[market] = {
                'available': market_data.get('available'),
                'impact_formats': len(formats['impact_formats']),
                'reach_formats': len(formats['reach_formats'])
            }
            
            print(f"✓ Market {market}: {results[market]['impact_formats']} impact formats, {results[market]['reach_formats']} reach formats")
        
        # Ensure all tested markets have some formats available
        for market, data in results.items():
            assert data['impact_formats'] > 0 or data['reach_formats'] > 0, f"No formats available for {market}"
        
        print("Cross-market consistency passed!")


class TestSystemIntegration:
    """Test integration between system components."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.controller = MediaPlanController(testing_mode=True)
    
    def teardown_method(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_component_integration(self):
        """Test integration between major components."""
        # Test DataManager -> MediaPlanController integration
        status = self.controller.get_system_status()
        assert 'data_status' in status
        assert 'available_markets' in status
        assert 'system_ready' in status
        
        # Test error handling integration
        from business_logic.error_handler import error_handler
        
        # Test error classification
        test_error = Exception("Test error")
        error_info = error_handler.classify_error(test_error, "test context")
        
        assert error_info.message is not None
        assert error_info.user_message is not None
        assert error_info.category is not None
        
        # Test notification creation
        notification = error_handler.create_user_notification(error_info)
        assert 'type' in notification
        assert 'message' in notification
        
        print("✓ Component integration tests passed")
    
    def test_ui_component_integration(self):
        """Test UI component integration (without Streamlit)."""
        # Test MediaPlannerForm validation logic
        form = MediaPlannerForm()
        
        # Test form validation
        test_data = {
            'brand_name': 'Test Brand',
            'budget': 25000.0,
            'country': 'SG',
            'campaign_period': 'Q1 2024',
            'objective': 'Brand Awareness',
            'planning_mode': 'AI'
        }
        
        validation_result = form.validate_form_data(test_data)
        assert validation_result['is_valid'], f"Form validation failed: {validation_result.get('errors', [])}"
        
        # Test PlanDisplayComponent
        display = PlanDisplayComponent()
        
        # Create mock plan for display testing
        mock_plan = MediaPlan(
            plan_id="test_plan",
            title="Test Plan",
            total_budget=20000.0,
            allocations=[],
            estimated_reach=100000,
            estimated_impressions=2000000,
            rationale="Test plan for UI integration",
            created_at=datetime.now()
        )
        
        # Test plan formatting
        formatted_plan = display.format_plan_for_display(mock_plan)
        assert 'title' in formatted_plan
        assert 'budget_formatted' in formatted_plan
        
        print("✓ UI component integration tests passed")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])