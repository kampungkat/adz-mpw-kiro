"""
Tests for AI plan generation system components.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from models.data_models import ClientBrief, MediaPlan, FormatAllocation
from business_logic.ai_plan_generator import AIPlanGenerator
from business_logic.budget_optimizer import BudgetOptimizer, OptimizationStrategy
from business_logic.plan_validator import PlanValidator, ValidationSeverity
from business_logic.media_plan_controller import MediaPlanController


class TestAIPlanGenerator:
    """Test cases for AIPlanGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_manager = Mock()
        self.generator = AIPlanGenerator(self.mock_data_manager, skip_openai_init=True)
        
        # Mock market data
        self.mock_market_data = {
            'available': True,
            'rate_card': {
                'impact_formats': {
                    'Display Banner': {'cpm': 2.50},
                    'Video Pre-roll': {'cpm': 8.00}
                },
                'reach_formats': {
                    'Native Content': {'cpm': 4.00}
                }
            },
            'sites': {
                'sites_by_format': {
                    'Display Banner': ['site1.com', 'site2.com'],
                    'Video Pre-roll': ['video1.com'],
                    'Native Content': ['native1.com', 'native2.com']
                }
            }
        }
        
        # Mock client brief
        self.client_brief = ClientBrief(
            brand_name="Test Brand",
            budget=10000.0,
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
    
    def test_create_system_prompt(self):
        """Test system prompt generation."""
        prompt = self.generator.create_system_prompt(self.mock_market_data, self.client_brief)
        
        assert "Test Brand" in prompt
        assert "$10,000.00" in prompt
        assert "SG" in prompt
        assert "Display Banner" in prompt
        assert "Video Pre-roll" in prompt
        assert "Native Content" in prompt
    
    @patch('business_logic.ai_plan_generator.OpenAI')
    def test_generate_multiple_plans_success(self, mock_openai_class):
        """Test successful plan generation."""
        # Mock OpenAI response
        mock_response = {
            'plans': [
                {
                    'title': 'Reach-Focused Strategy',
                    'rationale': 'Maximize unique audience reach',
                    'total_budget': 9500.0,
                    'estimated_reach': 50000,
                    'estimated_impressions': 1000000,
                    'allocations': [
                        {
                            'format_name': 'Display Banner',
                            'budget_allocation': 6000.0,
                            'cpm': 2.50,
                            'estimated_impressions': 2400000,
                            'recommended_sites': ['site1.com'],
                            'notes': 'High reach format'
                        },
                        {
                            'format_name': 'Native Content',
                            'budget_allocation': 3500.0,
                            'cpm': 4.00,
                            'estimated_impressions': 875000,
                            'recommended_sites': ['native1.com'],
                            'notes': 'Quality engagement'
                        }
                    ]
                }
            ]
        }
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps(mock_response)
        mock_client.chat.completions.create.return_value = mock_completion
        
        mock_openai_class.return_value = mock_client
        self.generator.client = mock_client
        
        # Mock data manager
        self.mock_data_manager.get_market_data.return_value = self.mock_market_data
        
        # Generate plans
        plans = self.generator.generate_multiple_plans(self.client_brief)
        
        assert len(plans) == 1
        assert isinstance(plans[0], MediaPlan)
        assert plans[0].title == 'Reach-Focused Strategy'
        assert plans[0].total_budget == 9500.0
        assert len(plans[0].allocations) == 2


class TestBudgetOptimizer:
    """Test cases for BudgetOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = BudgetOptimizer()
        
        self.client_brief = ClientBrief(
            brand_name="Test Brand",
            budget=10000.0,
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
        
        self.available_formats = {
            'Display Banner': {
                'cpm': 2.50,
                'sites': ['site1.com', 'site2.com']
            },
            'Video Pre-roll': {
                'cpm': 8.00,
                'sites': ['video1.com']
            },
            'Native Content': {
                'cpm': 4.00,
                'sites': ['native1.com', 'native2.com']
            }
        }
    
    def test_optimize_for_reach(self):
        """Test reach-focused optimization."""
        result = self.optimizer.optimize_budget_allocation(
            self.client_brief,
            self.available_formats,
            OptimizationStrategy.REACH_FOCUSED
        )
        
        assert result.total_budget_used > 0
        assert len(result.allocations) > 0
        assert result.estimated_total_reach > 0
        assert result.optimization_score >= 0
        
        # Check that lower CPM formats get higher allocation
        display_allocation = next(
            (alloc for alloc in result.allocations if alloc.format_name == 'Display Banner'), 
            None
        )
        video_allocation = next(
            (alloc for alloc in result.allocations if alloc.format_name == 'Video Pre-roll'), 
            None
        )
        
        if display_allocation and video_allocation:
            assert display_allocation.budget_allocation >= video_allocation.budget_allocation
    
    def test_optimize_for_frequency(self):
        """Test frequency-focused optimization."""
        result = self.optimizer.optimize_budget_allocation(
            self.client_brief,
            self.available_formats,
            OptimizationStrategy.FREQUENCY_FOCUSED
        )
        
        assert result.total_budget_used > 0
        assert len(result.allocations) <= 2  # Should focus on fewer formats
        assert result.average_frequency > 0
    
    def test_optimize_balanced(self):
        """Test balanced optimization."""
        result = self.optimizer.optimize_budget_allocation(
            self.client_brief,
            self.available_formats,
            OptimizationStrategy.BALANCED
        )
        
        assert result.total_budget_used > 0
        assert len(result.allocations) > 0
        assert result.optimization_score >= 0
        assert "Balanced" in result.strategy_notes


class TestPlanValidator:
    """Test cases for PlanValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PlanValidator()
        
        self.client_brief = ClientBrief(
            brand_name="Test Brand",
            budget=10000.0,
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
        
        self.available_formats = {
            'impact_formats': {
                'Display Banner': {'cpm': 2.50},
                'Video Pre-roll': {'cpm': 8.00}
            },
            'reach_formats': {
                'Native Content': {'cpm': 4.00}
            }
        }
        
        self.valid_plans_data = {
            'plans': [
                {
                    'title': 'Test Plan',
                    'rationale': 'Test rationale',
                    'total_budget': 9000.0,
                    'estimated_reach': 45000,
                    'estimated_impressions': 900000,
                    'allocations': [
                        {
                            'format_name': 'Display Banner',
                            'budget_allocation': 5000.0,
                            'cpm': 2.50,
                            'estimated_impressions': 2000000,
                            'recommended_sites': ['site1.com'],
                            'notes': 'Primary format'
                        },
                        {
                            'format_name': 'Native Content',
                            'budget_allocation': 4000.0,
                            'cpm': 4.00,
                            'estimated_impressions': 1000000,
                            'recommended_sites': ['native1.com'],
                            'notes': 'Secondary format'
                        }
                    ]
                }
            ]
        }
    
    def test_parse_valid_plans(self):
        """Test parsing of valid plans."""
        result = self.validator.parse_and_validate_plans(
            self.valid_plans_data,
            self.client_brief,
            self.available_formats
        )
        
        assert result.is_valid
        assert len(result.parsed_plans) == 1
        assert result.total_errors == 0
        
        plan = result.parsed_plans[0]
        assert plan.title == 'Test Plan'
        assert plan.total_budget == 9000.0
        assert len(plan.allocations) == 2
    
    def test_parse_invalid_json(self):
        """Test handling of invalid JSON."""
        invalid_json = "{ invalid json }"
        
        result = self.validator.parse_and_validate_plans(
            invalid_json,
            self.client_brief,
            self.available_formats
        )
        
        assert not result.is_valid
        assert result.total_errors > 0
        assert len(result.parsed_plans) == 0
    
    def test_budget_validation(self):
        """Test budget constraint validation."""
        # Create plan that exceeds budget
        over_budget_data = self.valid_plans_data.copy()
        over_budget_data['plans'][0]['total_budget'] = 15000.0  # Exceeds $10k budget
        
        result = self.validator.parse_and_validate_plans(
            over_budget_data,
            self.client_brief,
            self.available_formats
        )
        
        # Should have budget-related errors
        budget_errors = [
            issue for issue in result.issues 
            if 'budget' in issue.message.lower() and issue.severity == ValidationSeverity.ERROR
        ]
        assert len(budget_errors) > 0
    
    def test_missing_required_fields(self):
        """Test validation of missing required fields."""
        incomplete_data = {
            'plans': [
                {
                    'title': 'Incomplete Plan',
                    # Missing rationale, total_budget, allocations
                }
            ]
        }
        
        result = self.validator.parse_and_validate_plans(
            incomplete_data,
            self.client_brief,
            self.available_formats
        )
        
        assert not result.is_valid
        assert result.total_errors > 0
        assert len(result.parsed_plans) == 0


class TestMediaPlanController:
    """Test cases for MediaPlanController class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_manager = Mock()
        self.controller = MediaPlanController(self.mock_data_manager, testing_mode=True)
        
        self.client_brief = ClientBrief(
            brand_name="Test Brand",
            budget=10000.0,
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
    
    def test_validate_inputs_success(self):
        """Test successful input validation."""
        # Mock available markets
        self.mock_data_manager.get_available_markets.return_value = ['SG', 'MY', 'TH']
        
        is_valid, message = self.controller.validate_inputs(self.client_brief)
        
        assert is_valid
        assert "passed" in message.lower()
    
    def test_validate_inputs_missing_brand(self):
        """Test validation with missing brand name."""
        invalid_brief = ClientBrief(
            brand_name="",  # Empty brand name
            budget=10000.0,
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
        
        is_valid, message = self.controller.validate_inputs(invalid_brief)
        
        assert not is_valid
        assert "brand name" in message.lower()
    
    def test_validate_inputs_invalid_budget(self):
        """Test validation with invalid budget."""
        invalid_brief = ClientBrief(
            brand_name="Test Brand",
            budget=-1000.0,  # Negative budget
            country="SG",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
        
        is_valid, message = self.controller.validate_inputs(invalid_brief)
        
        assert not is_valid
        assert "budget" in message.lower()
    
    def test_validate_inputs_unavailable_market(self):
        """Test validation with unavailable market."""
        # Mock available markets (not including XX)
        self.mock_data_manager.get_available_markets.return_value = ['SG', 'MY', 'TH']
        
        invalid_brief = ClientBrief(
            brand_name="Test Brand",
            budget=10000.0,
            country="XX",  # Unavailable market
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
        
        is_valid, message = self.controller.validate_inputs(invalid_brief)
        
        assert not is_valid
        assert "not available" in message.lower()
    
    @patch('business_logic.media_plan_controller.AIPlanGenerator')
    def test_get_system_status(self, mock_ai_generator):
        """Test system status retrieval."""
        # Mock data manager responses
        self.mock_data_manager.validate_data_freshness.return_value = {
            'overall_status': 'ready'
        }
        self.mock_data_manager.get_available_markets.return_value = ['SG', 'MY']
        self.mock_data_manager.get_cache_stats.return_value = {
            'cache_dir_size': 1024
        }
        
        # Mock AI generator
        mock_ai_instance = Mock()
        mock_ai_instance.get_model_info.return_value = {
            'active_model': 'gpt-4'
        }
        mock_ai_generator.return_value = mock_ai_instance
        
        status = self.controller.get_system_status()
        
        assert 'data_status' in status
        assert 'available_markets' in status
        assert 'market_count' in status
        assert status['market_count'] == 2
        assert status['system_ready'] is True


if __name__ == '__main__':
    pytest.main([__file__])