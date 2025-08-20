"""
Unit tests for enhanced AIPlanGenerator with fine-tuning capabilities.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime

from business_logic.ai_plan_generator import AIPlanGenerator
from models.data_models import ClientBrief, MediaPlan, FormatAllocation


class TestAIPlanGeneratorEnhanced(unittest.TestCase):
    """Test cases for enhanced AIPlanGenerator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = AIPlanGenerator(skip_openai_init=True)
        
        self.test_client_brief = ClientBrief(
            brand_name="Test Brand",
            budget=50000.0,
            country="US",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI"
        )
        
        self.test_allocation = FormatAllocation(
            format_name="Display Banner",
            budget_allocation=25000.0,
            cpm=5.0,
            estimated_impressions=5000000,
            recommended_sites=["site1.com", "site2.com"],
            notes="High-impact placement"
        )
        
        self.test_media_plan = MediaPlan(
            plan_id="test_plan_1",
            title="Test Strategy",
            total_budget=50000.0,
            allocations=[self.test_allocation],
            estimated_reach=2000000,
            estimated_impressions=5000000,
            rationale="Test rationale for strategic approach",
            created_at=datetime.now()
        )
    
    def test_model_selection_strategy(self):
        """Test model selection strategy configuration."""
        # Test valid strategies
        valid_strategies = ['auto', 'base_only', 'fine_tuned_only']
        
        for strategy in valid_strategies:
            self.generator.set_model_selection_strategy(strategy)
            self.assertEqual(self.generator.model_selection_strategy, strategy)
        
        # Test invalid strategy
        with self.assertRaises(ValueError):
            self.generator.set_model_selection_strategy('invalid_strategy')
    
    def test_select_optimal_model_base_only(self):
        """Test model selection with base_only strategy."""
        self.generator.set_model_selection_strategy('base_only')
        self.generator.fine_tuned_model = 'ft:gpt-3.5-turbo:test'
        
        selected_model = self.generator.select_optimal_model(self.test_client_brief)
        self.assertEqual(selected_model, self.generator.model_name)
    
    def test_select_optimal_model_fine_tuned_only(self):
        """Test model selection with fine_tuned_only strategy."""
        self.generator.set_model_selection_strategy('fine_tuned_only')
        self.generator.fine_tuned_model = 'ft:gpt-3.5-turbo:test'
        
        selected_model = self.generator.select_optimal_model(self.test_client_brief)
        self.assertEqual(selected_model, 'ft:gpt-3.5-turbo:test')
    
    def test_select_optimal_model_fine_tuned_only_no_model(self):
        """Test model selection with fine_tuned_only strategy but no fine-tuned model."""
        self.generator.set_model_selection_strategy('fine_tuned_only')
        self.generator.fine_tuned_model = None
        
        selected_model = self.generator.select_optimal_model(self.test_client_brief)
        self.assertEqual(selected_model, self.generator.model_name)  # Falls back to base
    
    def test_auto_select_model_no_fine_tuned(self):
        """Test auto model selection when no fine-tuned model is available."""
        self.generator.set_model_selection_strategy('auto')
        self.generator.fine_tuned_model = None
        
        selected_model = self.generator.select_optimal_model(self.test_client_brief)
        self.assertEqual(selected_model, self.generator.model_name)
    
    def test_auto_select_model_insufficient_data(self):
        """Test auto model selection with insufficient performance data."""
        self.generator.set_model_selection_strategy('auto')
        self.generator.fine_tuned_model = 'ft:gpt-3.5-turbo:test'
        
        # No performance data yet
        selected_model = self.generator.select_optimal_model(self.test_client_brief)
        self.assertEqual(selected_model, 'ft:gpt-3.5-turbo:test')  # Prefers fine-tuned
    
    def test_auto_select_model_with_performance_data(self):
        """Test auto model selection with sufficient performance data."""
        self.generator.set_model_selection_strategy('auto')
        self.generator.fine_tuned_model = 'ft:gpt-3.5-turbo:test'
        
        # Add performance data favoring base model
        for i in range(10):
            self.generator.track_model_performance('gpt-4', 2.0, True, 0.9)
            self.generator.track_model_performance('ft:gpt-3.5-turbo:test', 1.5, True, 0.7)
        
        selected_model = self.generator.select_optimal_model(self.test_client_brief)
        self.assertEqual(selected_model, self.generator.model_name)  # Should prefer base model
    
    def test_track_model_performance(self):
        """Test model performance tracking."""
        # Track base model performance
        self.generator.track_model_performance('gpt-4', 2.5, True, 0.85)
        
        base_metrics = self.generator.performance_metrics['base_model']
        self.assertEqual(len(base_metrics['response_times']), 1)
        self.assertEqual(base_metrics['response_times'][0], 2.5)
        self.assertEqual(len(base_metrics['quality_scores']), 1)
        self.assertEqual(base_metrics['quality_scores'][0], 0.85)
        
        # Track fine-tuned model performance
        self.generator.track_model_performance('ft:gpt-3.5-turbo:test', 1.8, True, 0.90)
        
        ft_metrics = self.generator.performance_metrics['fine_tuned_model']
        self.assertEqual(len(ft_metrics['response_times']), 1)
        self.assertEqual(ft_metrics['response_times'][0], 1.8)
        self.assertEqual(len(ft_metrics['quality_scores']), 1)
        self.assertEqual(ft_metrics['quality_scores'][0], 0.90)
    
    def test_track_model_performance_success_rate(self):
        """Test success rate tracking in model performance."""
        # Track multiple calls with mixed success
        for i in range(5):
            self.generator.track_model_performance('gpt-4', 2.0, True)
        
        for i in range(3):
            self.generator.track_model_performance('gpt-4', 2.0, False)
        
        # Success rate should be 5/8 = 0.625
        base_metrics = self.generator.performance_metrics['base_model']
        self.assertAlmostEqual(base_metrics['success_rate'], 0.625, places=3)
    
    def test_track_model_cost(self):
        """Test model cost tracking."""
        # Track base model cost
        self.generator.track_model_cost('gpt-4', 1000, 500)
        
        self.assertEqual(self.generator.cost_tracker['base_model_calls'], 1)
        self.assertEqual(self.generator.cost_tracker['base_model_tokens'], 1500)
        self.assertGreater(self.generator.cost_tracker['total_cost'], 0)
        
        # Track fine-tuned model cost
        self.generator.track_model_cost('ft:gpt-3.5-turbo:test', 1000, 500)
        
        self.assertEqual(self.generator.cost_tracker['fine_tuned_model_calls'], 1)
        self.assertEqual(self.generator.cost_tracker['fine_tuned_model_tokens'], 1500)
    
    def test_get_cost_analysis(self):
        """Test cost analysis generation."""
        # Add some cost data
        self.generator.track_model_cost('gpt-4', 1000, 500)
        self.generator.track_model_cost('ft:gpt-3.5-turbo:test', 1000, 500)
        
        analysis = self.generator.get_cost_analysis()
        
        self.assertIn('total_cost', analysis)
        self.assertIn('base_model', analysis)
        self.assertIn('fine_tuned_model', analysis)
        self.assertIn('recommendations', analysis)
        
        # Check structure
        self.assertIn('calls', analysis['base_model'])
        self.assertIn('cost_per_call', analysis['base_model'])
        self.assertIn('calls', analysis['fine_tuned_model'])
        self.assertIn('cost_per_call', analysis['fine_tuned_model'])
    
    def test_get_performance_summary(self):
        """Test performance summary generation."""
        # Add performance data
        for i in range(5):
            self.generator.track_model_performance('gpt-4', 2.0 + i * 0.1, True, 0.8 + i * 0.02)
            self.generator.track_model_performance('ft:gpt-3.5-turbo:test', 1.5 + i * 0.1, True, 0.85 + i * 0.02)
        
        summary = self.generator._get_performance_summary()
        
        self.assertIn('base_model', summary)
        self.assertIn('fine_tuned_model', summary)
        
        # Check base model summary
        base_summary = summary['base_model']
        self.assertEqual(base_summary['total_calls'], 5)
        self.assertEqual(base_summary['success_rate'], 1.0)
        self.assertGreater(base_summary['avg_response_time'], 0)
        self.assertGreater(base_summary['avg_quality_score'], 0)
        
        # Check fine-tuned model summary
        ft_summary = summary['fine_tuned_model']
        self.assertEqual(ft_summary['total_calls'], 5)
        self.assertEqual(ft_summary['success_rate'], 1.0)
        self.assertGreater(ft_summary['avg_response_time'], 0)
        self.assertGreater(ft_summary['avg_quality_score'], 0)
    
    def test_evaluate_plan_quality(self):
        """Test plan quality evaluation."""
        quality_score = self.generator.evaluate_plan_quality(
            self.test_media_plan, 
            self.test_client_brief
        )
        
        # Should be a score between 0 and 1
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
        
        # Should be relatively high for a well-formed plan
        self.assertGreater(quality_score, 0.5)
    
    def test_evaluate_plan_quality_over_budget(self):
        """Test plan quality evaluation for over-budget plan."""
        over_budget_plan = MediaPlan(
            plan_id="over_budget",
            title="Over Budget Plan",
            total_budget=60000.0,  # Exceeds client budget of 50000
            allocations=[self.test_allocation],
            estimated_reach=2000000,
            estimated_impressions=5000000,
            rationale="Test rationale",
            created_at=datetime.now()
        )
        
        quality_score = self.generator.evaluate_plan_quality(
            over_budget_plan, 
            self.test_client_brief
        )
        
        # Should have lower quality due to budget violation
        normal_quality = self.generator.evaluate_plan_quality(
            self.test_media_plan, 
            self.test_client_brief
        )
        
        self.assertLess(quality_score, normal_quality)
    
    def test_evaluate_plan_quality_no_allocations(self):
        """Test plan quality evaluation for plan with no allocations."""
        empty_plan = MediaPlan(
            plan_id="empty",
            title="Empty Plan",
            total_budget=0.0,
            allocations=[],
            estimated_reach=0,
            estimated_impressions=0,
            rationale="",
            created_at=datetime.now()
        )
        
        quality_score = self.generator.evaluate_plan_quality(
            empty_plan, 
            self.test_client_brief
        )
        
        # Should have very low quality
        self.assertLess(quality_score, 0.3)
    
    def test_reset_performance_metrics(self):
        """Test resetting performance metrics."""
        # Add some data
        self.generator.track_model_performance('gpt-4', 2.0, True, 0.8)
        self.generator.track_model_cost('gpt-4', 1000, 500)
        
        # Verify data exists
        self.assertGreater(len(self.generator.performance_metrics['base_model']['response_times']), 0)
        self.assertGreater(self.generator.cost_tracker['base_model_calls'], 0)
        
        # Reset
        self.generator.reset_performance_metrics()
        
        # Verify data is cleared
        self.assertEqual(len(self.generator.performance_metrics['base_model']['response_times']), 0)
        self.assertEqual(self.generator.cost_tracker['base_model_calls'], 0)
        self.assertEqual(self.generator.cost_tracker['total_cost'], 0.0)
    
    def test_get_model_info_enhanced(self):
        """Test enhanced model info with performance and cost data."""
        # Add some performance data
        self.generator.track_model_performance('gpt-4', 2.0, True, 0.8)
        self.generator.track_model_cost('gpt-4', 1000, 500)
        
        model_info = self.generator.get_model_info()
        
        # Check enhanced fields
        self.assertIn('selection_strategy', model_info)
        self.assertIn('cost_tracker', model_info)
        self.assertIn('performance_summary', model_info)
        
        # Verify cost tracker data
        self.assertEqual(model_info['cost_tracker']['base_model_calls'], 1)
        
        # Verify performance summary
        self.assertIn('base_model', model_info['performance_summary'])
        self.assertEqual(model_info['performance_summary']['base_model']['total_calls'], 1)


if __name__ == '__main__':
    unittest.main()