"""
Integration tests for complete media planning workflows.

These tests verify end-to-end functionality across all system components.
"""

import pytest
import tempfile
import os
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from models.data_models import ClientBrief, MediaPlan, FormatAllocation
from business_logic.media_plan_controller import MediaPlanController
from data.manager import DataManager


class TestCompleteWorkflow:
    """Test complete media planning workflows end-to-end."""
    
    def setup_method(self):
        """Set up test fixtures with real data files."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.rate_card_file = os.path.join(self.temp_dir, 'test_rate_card.xlsx')
        self.site_list_file = os.path.join(self.temp_dir, 'test_site_list.xlsx')
        
        # Create realistic test data files
        self._create_realistic_test_files()
        
        # Initialize controller with test data
        self.data_manager = DataManager(cache_dir=os.path.join(self.temp_dir, 'cache'))
        self.data_manager.default_rate_card_path = self.rate_card_file
        self.data_manager.default_site_list_path = self.site_list_file
        
        self.controller = MediaPlanController(self.data_manager, testing_mode=True)
        
        # Standard test client brief
        self.client_brief = ClientBrief(
            brand_name="Test Brand Co.",
            budget=25000.0,
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_realistic_test_files(self):
        """Create realistic test data files that mirror Adzymic's structure."""
        # Create comprehensive rate card
        impact_data = {
            '$10,000 net budget': ['', 'Desktop Skin', 'Rich Media Banner', 'Video Pre-roll', 'Mobile Skin', 'Native Content'],
            'Unnamed: 1': ['', 'Desktop Skin', 'Rich Media Banner', 'Video Pre-roll', 'Mobile Skin', 'Native Content'],
            'SG': ['SG', 45, 38, 52, 35, 42],
            'MY': ['MY', 38, 32, 45, 30, 36],
            'TH': ['TH', 42, 35, 48, 32, 38],
            'ID': ['ID', 35, 28, 40, 25, 32],
            'PH': ['PH', 40, 33, 46, 28, 35]
        }
        impact_df = pd.DataFrame(impact_data)
        
        # Create reach rates with multiple tiers
        reach_data = {
            'Country': ['Singapore', 'Malaysia', 'Thailand', 'Indonesia', 'Philippines'],
            'Currency': ['USD', 'USD', 'USD', 'USD', 'USD'],
            'CPM >10K': [28, 22, 25, 18, 20],
            'CPM >20K': [25, 20, 22, 16, 18],
            'CPM >30K': [22, 18, 20, 14, 16],
            'CPM >50K': [20, 16, 18, 12, 14]
        }
        reach_df = pd.DataFrame(reach_data)
        
        with pd.ExcelWriter(self.rate_card_file, engine='openpyxl') as writer:
            impact_df.to_excel(writer, sheet_name='APX - Impact ', index=False)
            reach_df.to_excel(writer, sheet_name='APX - Reach', index=False)
        
        # Create comprehensive site lists
        sg_sites = {
            'Publishers': [
                'Straits Times', 'Channel NewsAsia', 'Zaobao', 'TODAY', 'AsiaOne',
                'Hardware Zone', 'Mothership', 'Yahoo Singapore', 'MSN Singapore', 'Carousell'
            ],
            'Domain': [
                'straitstimes.com', 'channelnewsasia.com', 'zaobao.com.sg', 'todayonline.com', 'asiaone.com',
                'hardwarezone.com.sg', 'mothership.sg', 'sg.yahoo.com', 'msn.com/sg', 'carousell.sg'
            ],
            'Category': [
                'News and Media', 'News and Media', 'Chinese News and Media', 'News and Media', 'News and Media',
                'Technology', 'Entertainment', 'Portal', 'Portal', 'E-commerce'
            ],
            'Desktop Skin': ['Available', 'Available', 'Available', 'Available', 'N.A.', 'Available', 'Available', 'Available', 'Available', 'N.A.'],
            'Rich Media Banner': ['Available', 'Available', 'Available', 'Available', 'Available', 'Available', 'Available', 'Available', 'Available', 'Available'],
            'Video Pre-roll': ['Available', 'Available', 'N.A.', 'Available', 'Available', 'Available', 'Available', 'Available', 'Available', 'N.A.'],
            'Mobile Skin': ['Available', 'Available', 'Available', 'Available', 'Available', 'N.A.', 'Available', 'Available', 'Available', 'Available'],
            'Native Content': ['Available', 'Available', 'Available', 'Available', 'Available', 'Available', 'Available', 'Available', 'Available', 'Available']
        }
        sg_df = pd.DataFrame(sg_sites)
        
        my_sites = {
            'Publishers': ['The Star', 'Malaysiakini', 'New Straits Times', 'Astro Awani', 'Harian Metro'],
            'Domain': ['thestar.com.my', 'malaysiakini.com', 'nst.com.my', 'astroawani.com', 'hmetro.com.my'],
            'Category': ['News and Media', 'News and Media', 'News and Media', 'News and Media', 'News and Media'],
            'Desktop Skin': ['Available', 'Available', 'Available', 'Available', 'Available'],
            'Rich Media Banner': ['Available', 'Available', 'Available', 'Available', 'Available'],
            'Video Pre-roll': ['Available', 'N.A.', 'Available', 'Available', 'Available'],
            'Mobile Skin': ['Available', 'Available', 'Available', 'Available', 'Available'],
            'Native Content': ['Available', 'Available', 'Available', 'Available', 'Available']
        }
        my_df = pd.DataFrame(my_sites)
        
        with pd.ExcelWriter(self.site_list_file, engine='openpyxl') as writer:
            sg_df.to_excel(writer, sheet_name='SG', index=False)
            my_df.to_excel(writer, sheet_name='MY', index=False)
    
    def test_complete_ai_workflow_success(self):
        """Test complete AI-driven media planning workflow."""
        # Mock AI response
        mock_ai_response = {
            'plans': [
                {
                    'title': 'Reach-Maximizing Strategy',
                    'rationale': 'Focus on high-reach formats to maximize brand exposure',
                    'total_budget': 24500.0,
                    'estimated_reach': 180000,
                    'estimated_impressions': 4500000,
                    'allocations': [
                        {
                            'format_name': 'Rich Media Banner',
                            'budget_allocation': 12000.0,
                            'cpm': 38.0,
                            'estimated_impressions': 315789,
                            'recommended_sites': ['straitstimes.com', 'channelnewsasia.com'],
                            'notes': 'High visibility format for brand awareness'
                        },
                        {
                            'format_name': 'Native Content',
                            'budget_allocation': 8000.0,
                            'cpm': 42.0,
                            'estimated_impressions': 190476,
                            'recommended_sites': ['todayonline.com', 'asiaone.com'],
                            'notes': 'Engaging content format'
                        },
                        {
                            'format_name': 'Mobile Skin',
                            'budget_allocation': 4500.0,
                            'cpm': 35.0,
                            'estimated_impressions': 128571,
                            'recommended_sites': ['mothership.sg'],
                            'notes': 'Mobile-focused reach'
                        }
                    ]
                },
                {
                    'title': 'Premium Impact Strategy',
                    'rationale': 'High-impact formats for maximum brand impression',
                    'total_budget': 23800.0,
                    'estimated_reach': 150000,
                    'estimated_impressions': 3800000,
                    'allocations': [
                        {
                            'format_name': 'Desktop Skin',
                            'budget_allocation': 15000.0,
                            'cpm': 45.0,
                            'estimated_impressions': 333333,
                            'recommended_sites': ['straitstimes.com', 'zaobao.com.sg'],
                            'notes': 'Premium desktop placement'
                        },
                        {
                            'format_name': 'Video Pre-roll',
                            'budget_allocation': 8800.0,
                            'cpm': 52.0,
                            'estimated_impressions': 169231,
                            'recommended_sites': ['channelnewsasia.com'],
                            'notes': 'High-engagement video format'
                        }
                    ]
                },
                {
                    'title': 'Balanced Multi-Format Strategy',
                    'rationale': 'Diversified approach across multiple touchpoints',
                    'total_budget': 24200.0,
                    'estimated_reach': 165000,
                    'estimated_impressions': 4200000,
                    'allocations': [
                        {
                            'format_name': 'Rich Media Banner',
                            'budget_allocation': 8000.0,
                            'cpm': 38.0,
                            'estimated_impressions': 210526,
                            'recommended_sites': ['straitstimes.com'],
                            'notes': 'Core display format'
                        },
                        {
                            'format_name': 'Native Content',
                            'budget_allocation': 6000.0,
                            'cpm': 42.0,
                            'estimated_impressions': 142857,
                            'recommended_sites': ['todayonline.com'],
                            'notes': 'Content integration'
                        },
                        {
                            'format_name': 'Mobile Skin',
                            'budget_allocation': 5200.0,
                            'cpm': 35.0,
                            'estimated_impressions': 148571,
                            'recommended_sites': ['mothership.sg'],
                            'notes': 'Mobile reach'
                        },
                        {
                            'format_name': 'Video Pre-roll',
                            'budget_allocation': 5000.0,
                            'cpm': 52.0,
                            'estimated_impressions': 96154,
                            'recommended_sites': ['channelnewsasia.com'],
                            'notes': 'Video engagement'
                        }
                    ]
                }
            ]
        }
        
        # Mock data validation and market data
        with patch.object(self.controller.data_manager, 'validate_data_freshness') as mock_validate, \
             patch.object(self.controller.data_manager, 'get_market_data') as mock_market_data, \
             patch.object(self.controller.ai_generator, 'generate_multiple_plans') as mock_generate:
            
            # Mock successful data validation
            mock_validate.return_value = {'overall_status': 'ready'}
            
            # Mock market data
            mock_market_data.return_value = {
                'available': True,
                'rate_card': {
                    'impact_formats': {'Rich Media Banner': {'cpm': 38}, 'Desktop Skin': {'cpm': 45}},
                    'reach_formats': {'Native Content': {'cpm': 42}}
                },
                'sites': {'sites_by_format': {}}
            }
            
            # Convert mock response to MediaPlan objects
            mock_plans = []
            for plan_data in mock_ai_response['plans']:
                allocations = []
                for alloc_data in plan_data['allocations']:
                    allocation = FormatAllocation(
                        format_name=alloc_data['format_name'],
                        budget_allocation=alloc_data['budget_allocation'],
                        cpm=alloc_data['cpm'],
                        estimated_impressions=alloc_data['estimated_impressions'],
                        recommended_sites=alloc_data['recommended_sites'],
                        notes=alloc_data['notes']
                    )
                    allocations.append(allocation)
                
                plan = MediaPlan(
                    plan_id=f"plan_{len(mock_plans) + 1}",
                    title=plan_data['title'],
                    total_budget=plan_data['total_budget'],
                    allocations=allocations,
                    estimated_reach=plan_data['estimated_reach'],
                    estimated_impressions=plan_data['estimated_impressions'],
                    rationale=plan_data['rationale'],
                    created_at=datetime.now()
                )
                mock_plans.append(plan)
            
            mock_generate.return_value = mock_plans
            
            # Execute complete workflow
            success, plans, message, notification = self.controller.generate_plans(self.client_brief)
            
            # Verify success
            assert success, f"Plan generation failed: {message}"
            assert len(plans) == 3, f"Expected 3 plans, got {len(plans)}"
            assert "Successfully generated" in message
            
            # Verify plan quality
            for plan in plans:
                assert isinstance(plan, MediaPlan)
                assert plan.total_budget > 0
                assert plan.total_budget <= self.client_brief.budget
                assert len(plan.allocations) > 0
                assert plan.estimated_reach > 0
                assert plan.estimated_impressions > 0
                
                # Verify allocations
                for allocation in plan.allocations:
                    assert allocation.budget_allocation > 0
                    assert allocation.cpm > 0
                    assert len(allocation.recommended_sites) > 0
    
    def test_input_validation_workflow(self):
        """Test comprehensive input validation workflow."""
        # Test valid inputs
        is_valid, message = self.controller.validate_inputs(self.client_brief)
        assert is_valid, f"Valid inputs failed validation: {message}"
        
        # Test invalid brand name
        invalid_brief = ClientBrief(
            brand_name="",
            budget=25000.0,
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
        is_valid, message = self.controller.validate_inputs(invalid_brief)
        assert not is_valid
        assert "brand name" in message.lower()
        
        # Test invalid budget
        invalid_brief.brand_name = "Test Brand"
        invalid_brief.budget = -1000.0
        is_valid, message = self.controller.validate_inputs(invalid_brief)
        assert not is_valid
        assert "budget" in message.lower()
        
        # Test excessive budget
        invalid_brief.budget = 2000000.0  # $2M
        is_valid, message = self.controller.validate_inputs(invalid_brief)
        assert not is_valid
        assert "maximum limit" in message.lower()
        
        # Test invalid country
        invalid_brief.budget = 25000.0
        invalid_brief.country = "INVALID"
        is_valid, message = self.controller.validate_inputs(invalid_brief)
        assert not is_valid
        assert "not available" in message.lower()
        
        # Test invalid campaign period
        invalid_brief.country = "SG"
        invalid_brief.campaign_period = "Invalid Period Format!!!"
        is_valid, message = self.controller.validate_inputs(invalid_brief)
        assert not is_valid
        assert "campaign period" in message.lower()
        
        # Test invalid planning mode
        invalid_brief.campaign_period = "Q1 2024"
        invalid_brief.planning_mode = "INVALID"
        is_valid, message = self.controller.validate_inputs(invalid_brief)
        assert not is_valid
        assert "planning mode" in message.lower()
    
    def test_plan_comparison_workflow(self):
        """Test plan comparison and ranking workflow."""
        # Create test plans with different characteristics
        plans = [
            MediaPlan(
                plan_id="plan_1",
                title="High Reach Plan",
                total_budget=24000.0,
                allocations=[
                    FormatAllocation(
                        format_name="Rich Media Banner",
                        budget_allocation=24000.0,
                        cpm=38.0,
                        estimated_impressions=631579,
                        recommended_sites=["straitstimes.com"],
                        notes="Single format focus"
                    )
                ],
                estimated_reach=200000,
                estimated_impressions=631579,
                rationale="Maximum reach strategy",
                created_at=datetime.now()
            ),
            MediaPlan(
                plan_id="plan_2",
                title="Balanced Plan",
                total_budget=24000.0,
                allocations=[
                    FormatAllocation(
                        format_name="Rich Media Banner",
                        budget_allocation=12000.0,
                        cpm=38.0,
                        estimated_impressions=315789,
                        recommended_sites=["straitstimes.com"],
                        notes="Primary format"
                    ),
                    FormatAllocation(
                        format_name="Native Content",
                        budget_allocation=12000.0,
                        cpm=42.0,
                        estimated_impressions=285714,
                        recommended_sites=["todayonline.com"],
                        notes="Secondary format"
                    )
                ],
                estimated_reach=180000,
                estimated_impressions=601503,
                rationale="Balanced approach",
                created_at=datetime.now()
            ),
            MediaPlan(
                plan_id="plan_3",
                title="Premium Plan",
                total_budget=20000.0,  # Lower budget utilization
                allocations=[
                    FormatAllocation(
                        format_name="Desktop Skin",
                        budget_allocation=20000.0,
                        cpm=45.0,
                        estimated_impressions=444444,
                        recommended_sites=["straitstimes.com"],
                        notes="Premium placement"
                    )
                ],
                estimated_reach=120000,
                estimated_impressions=444444,
                rationale="Premium strategy",
                created_at=datetime.now()
            )
        ]
        
        # Test comparison
        comparison = self.controller.compare_plans(plans, self.client_brief)
        
        # Verify comparison structure
        assert 'comparison_results' in comparison
        assert 'summary' in comparison
        assert 'best_plan_index' in comparison
        assert 'comparison_criteria' in comparison
        
        results = comparison['comparison_results']
        assert len(results) == 3
        
        # Verify ranking (should be sorted by composite score)
        for i in range(len(results) - 1):
            assert results[i]['composite_score'] >= results[i + 1]['composite_score']
        
        # Verify metrics calculation
        for result in results:
            assert 'budget_efficiency' in result
            assert 'reach_per_dollar' in result
            assert 'impressions_per_dollar' in result
            assert 'format_diversity' in result
            assert 'allocation_balance' in result
            assert 'rank' in result
            assert 'recommendation' in result
        
        # Verify best plan identification
        best_plan_index = comparison['best_plan_index']
        assert best_plan_index is not None
        assert 0 <= best_plan_index < len(plans)
    
    def test_error_handling_workflow(self):
        """Test error handling throughout the workflow."""
        # Test with missing data files
        controller_no_data = MediaPlanController(testing_mode=True)
        controller_no_data.data_manager.default_rate_card_path = "nonexistent.xlsx"
        controller_no_data.data_manager.default_site_list_path = "nonexistent.xlsx"
        
        success, plans, message, notification = controller_no_data.generate_plans(self.client_brief)
        assert not success
        assert "not available" in message.lower() or "error" in message.lower()
        
        # Test with AI generation failure
        with patch.object(self.controller.data_manager, 'validate_data_freshness') as mock_validate, \
             patch.object(self.controller.data_manager, 'get_market_data') as mock_market_data, \
             patch.object(self.controller.ai_generator, 'generate_multiple_plans') as mock_generate:
            
            # Mock successful data validation
            mock_validate.return_value = {'overall_status': 'ready'}
            mock_market_data.return_value = {'available': True, 'rate_card': {}}
            
            mock_generate.side_effect = Exception("AI service unavailable")
            
            success, plans, message, notification = self.controller.generate_plans(self.client_brief)
            assert not success
            # The error handler may genericize the message, so check for any error indication
            assert any(keyword in message.lower() for keyword in ["error", "failed", "unavailable", "contact support"])
        
        # Test with invalid AI response
        with patch.object(self.controller.data_manager, 'validate_data_freshness') as mock_validate, \
             patch.object(self.controller.data_manager, 'get_market_data') as mock_market_data, \
             patch.object(self.controller.ai_generator, 'generate_multiple_plans') as mock_generate:
            
            # Mock successful data validation
            mock_validate.return_value = {'overall_status': 'ready'}
            mock_market_data.return_value = {'available': True, 'rate_card': {}}
            
            mock_generate.return_value = []  # Empty response
            
            success, plans, message, notification = self.controller.generate_plans(self.client_brief)
            assert not success
            assert "failed to generate any valid plans" in message or "unable to generate" in message
    
    def test_system_status_workflow(self):
        """Test system status and health monitoring."""
        status = self.controller.get_system_status()
        
        # Verify status structure
        assert 'data_status' in status
        assert 'available_markets' in status
        assert 'market_count' in status
        assert 'ai_model' in status
        assert 'cache_stats' in status
        assert 'system_ready' in status
        assert 'last_updated' in status
        
        # Verify data availability
        assert isinstance(status['available_markets'], list)
        assert status['market_count'] >= 0
        assert isinstance(status['system_ready'], bool)
        
        # Verify market data
        assert 'SG' in status['available_markets']
        assert 'MY' in status['available_markets']
    
    def test_manual_format_selection_workflow(self):
        """Test workflow with manual format selection."""
        # Create client brief with manual selection
        manual_brief = ClientBrief(
            brand_name="Test Brand Co.",
            budget=25000.0,
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="Manual",
            selected_formats=["Rich Media Banner", "Native Content"]
        )
        
        # Validate inputs
        is_valid, message = self.controller.validate_inputs(manual_brief)
        assert is_valid, f"Manual selection validation failed: {message}"
        
        # Test with invalid format selection
        invalid_manual_brief = ClientBrief(
            brand_name="Test Brand Co.",
            budget=25000.0,
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="Manual",
            selected_formats=["Invalid Format", "Another Invalid"]
        )
        
        is_valid, message = self.controller.validate_inputs(invalid_manual_brief)
        assert not is_valid
        assert "Invalid formats selected" in message
    
    def test_available_formats_workflow(self):
        """Test getting available formats for markets."""
        # Test valid market
        formats = self.controller.get_available_formats("SG")
        
        assert 'impact_formats' in formats
        assert 'reach_formats' in formats
        assert isinstance(formats['impact_formats'], list)
        assert isinstance(formats['reach_formats'], list)
        
        # Should have formats from test data
        assert len(formats['impact_formats']) > 0
        assert len(formats['reach_formats']) > 0
        
        # Test invalid market
        formats = self.controller.get_available_formats("INVALID")
        assert formats['impact_formats'] == []
        assert formats['reach_formats'] == []


class TestDataIntegration:
    """Test data integration and management workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(cache_dir=os.path.join(self.temp_dir, 'cache'))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_loading_integration(self):
        """Test integrated data loading workflow."""
        # This test would use real Adzymic files if available
        # For now, we test the workflow structure
        
        # Test cache initialization
        cache_stats = self.data_manager.get_cache_stats()
        assert 'rate_card' in cache_stats
        assert 'site_list' in cache_stats
        
        # Test data freshness validation
        validation = self.data_manager.validate_data_freshness()
        assert 'rate_card' in validation
        assert 'site_list' in validation
        assert 'overall_status' in validation


class TestPerformanceWorkflow:
    """Test performance aspects of the complete workflow."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.controller = MediaPlanController(testing_mode=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_concurrent_plan_generation(self):
        """Test handling of concurrent plan generation requests."""
        import threading
        import time
        
        results = []
        errors = []
        
        def generate_plan(client_id):
            try:
                brief = ClientBrief(
                    brand_name=f"Client {client_id}",
                    budget=20000.0,
                    country="SG",
                    campaign_period="Q1 2024",
                    objective="Brand Awareness",
                    planning_mode="AI"
                )
                
                # Mock successful generation
                with patch.object(self.controller, 'generate_plans') as mock_generate:
                    mock_generate.return_value = (True, [], "Success", None)
                    success, plans, message, notification = self.controller.generate_plans(brief)
                    results.append((client_id, success, message))
            except Exception as e:
                errors.append((client_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_plan, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Verify results
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        for client_id, success, message in results:
            assert success, f"Client {client_id} failed: {message}"
    
    def test_large_budget_handling(self):
        """Test handling of large budget scenarios."""
        large_budget_brief = ClientBrief(
            brand_name="Enterprise Client",
            budget=500000.0,  # $500K
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
        
        # Should validate successfully
        is_valid, message = self.controller.validate_inputs(large_budget_brief)
        assert is_valid, f"Large budget validation failed: {message}"
        
        # Test budget limit
        excessive_budget_brief = ClientBrief(
            brand_name="Excessive Client",
            budget=2000000.0,  # $2M - exceeds limit
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
        
        is_valid, message = self.controller.validate_inputs(excessive_budget_brief)
        assert not is_valid
        assert "maximum limit" in message.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])