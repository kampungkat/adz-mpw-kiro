"""
Tests for the PlanDisplayComponent and PlanExportComponent.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import streamlit as st

from ui.components import PlanDisplayComponent, PlanExportComponent
from models.data_models import MediaPlan, FormatAllocation


@pytest.fixture
def sample_plans():
    """Create sample MediaPlan objects for testing."""
    allocation1 = FormatAllocation(
        format_name="Display Banner",
        budget_allocation=15000.0,
        cpm=5.50,
        estimated_impressions=2727272,
        recommended_sites=["Site A", "Site B", "Site C"],
        notes="High-impact placements for brand awareness"
    )
    
    allocation2 = FormatAllocation(
        format_name="Video Pre-roll",
        budget_allocation=10000.0,
        cpm=8.00,
        estimated_impressions=1250000,
        recommended_sites=["Video Site 1", "Video Site 2"],
        notes="Premium video inventory for engagement"
    )
    
    plan1 = MediaPlan(
        plan_id="plan_1_test",
        title="Reach-Focused Strategy",
        total_budget=25000.0,
        allocations=[allocation1, allocation2],
        estimated_reach=1500000,
        estimated_impressions=3977272,
        rationale="This plan maximizes reach through cost-effective display placements combined with high-impact video advertising.",
        created_at=datetime.now()
    )
    
    allocation3 = FormatAllocation(
        format_name="Native Advertising",
        budget_allocation=20000.0,
        cpm=12.00,
        estimated_impressions=1666666,
        recommended_sites=["Premium Site 1", "Premium Site 2"],
        notes="Premium native placements for quality engagement"
    )
    
    plan2 = MediaPlan(
        plan_id="plan_2_test",
        title="Quality-Focused Strategy",
        total_budget=20000.0,
        allocations=[allocation3],
        estimated_reach=800000,
        estimated_impressions=1666666,
        rationale="This plan focuses on high-quality native advertising for better engagement rates and brand safety.",
        created_at=datetime.now()
    )
    
    return [plan1, plan2]


@pytest.fixture
def sample_client_brief():
    """Create sample client brief for testing."""
    return {
        'brand_name': 'Test Brand',
        'budget': 30000.0,
        'country': 'SG',
        'objective': 'Brand Awareness',
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',
        'planning_mode': 'AI Selection'
    }


class TestPlanDisplayComponent:
    """Test cases for PlanDisplayComponent."""
    
    def test_init(self):
        """Test component initialization."""
        component = PlanDisplayComponent()
        assert component.data_manager is None
        
        mock_data_manager = Mock()
        component = PlanDisplayComponent(mock_data_manager)
        assert component.data_manager == mock_data_manager
    
    @patch('streamlit.subheader')
    @patch('streamlit.warning')
    def test_render_plan_comparison_no_plans(self, mock_warning, mock_subheader):
        """Test rendering with no plans."""
        component = PlanDisplayComponent()
        result = component.render_plan_comparison([], {})
        
        mock_warning.assert_called_once_with("No media plans available to display.")
        assert result == {}
    
    def test_calculate_plan_score(self, sample_plans, sample_client_brief):
        """Test plan scoring calculation."""
        component = PlanDisplayComponent()
        plan = sample_plans[0]
        
        scores = component._calculate_plan_score(plan, sample_client_brief)
        
        # Check that all expected score categories are present
        expected_categories = ['Cost Efficiency', 'Reach Efficiency', 'Format Diversity', 'Budget Utilization', 'Overall Score']
        for category in expected_categories:
            assert category in scores
            assert 0 <= scores[category] <= 100
        
        # Check that overall score is average of other scores
        other_scores = [v for k, v in scores.items() if k != 'Overall Score']
        expected_overall = sum(other_scores) / len(other_scores)
        assert abs(scores['Overall Score'] - expected_overall) < 0.01
    
    def test_generate_optimization_recommendations(self, sample_plans, sample_client_brief):
        """Test optimization recommendations generation."""
        component = PlanDisplayComponent()
        plan = sample_plans[0]
        
        recommendations = component._generate_optimization_recommendations(
            plan, sample_plans, 'brand awareness', 30000.0
        )
        
        # Check that recommendations dictionary is returned
        assert isinstance(recommendations, dict)
        
        # Check for expected recommendation types
        possible_types = ['strength', 'improvement', 'alternative']
        for rec_type, rec_text in recommendations.items():
            assert rec_type in possible_types
            if rec_text:
                assert isinstance(rec_text, str)
                assert len(rec_text) > 0
    
    def test_analyze_format_mix(self, sample_plans):
        """Test format mix analysis."""
        component = PlanDisplayComponent()
        plan = sample_plans[0]
        
        insights = component._analyze_format_mix(plan, sample_plans)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Check that insights are strings
        for insight in insights:
            assert isinstance(insight, str)
            assert len(insight) > 0


class TestPlanExportComponent:
    """Test cases for PlanExportComponent."""
    
    def test_init(self):
        """Test component initialization."""
        component = PlanExportComponent()
        assert component.data_manager is None
        
        mock_data_manager = Mock()
        component = PlanExportComponent(mock_data_manager)
        assert component.data_manager == mock_data_manager
    
    def test_create_single_plan_csv(self, sample_plans):
        """Test CSV creation for single plan."""
        component = PlanExportComponent()
        plan = sample_plans[0]
        
        options = {
            'include_rationale': True,
            'include_sites': True,
            'include_metrics': True
        }
        
        csv_lines = component._create_single_plan_csv(plan, options)
        
        assert isinstance(csv_lines, list)
        assert len(csv_lines) > 0
        
        # Check for expected content
        csv_content = '\n'.join(csv_lines)
        assert plan.title in csv_content
        assert f"${plan.total_budget:,.2f}" in csv_content
        assert plan.rationale in csv_content
        
        # Check for allocation data
        for allocation in plan.allocations:
            assert allocation.format_name in csv_content
    
    def test_create_comparison_csv(self, sample_plans):
        """Test CSV creation for plan comparison."""
        component = PlanExportComponent()
        
        options = {
            'include_rationale': True,
            'include_sites': True,
            'include_metrics': True
        }
        
        csv_lines = component._create_comparison_csv(sample_plans, options)
        
        assert isinstance(csv_lines, list)
        assert len(csv_lines) > 0
        
        # Check for expected content
        csv_content = '\n'.join(csv_lines)
        assert "PLAN COMPARISON SUMMARY" in csv_content
        
        # Check that both plans are included
        for plan in sample_plans:
            assert plan.title in csv_content
    
    def test_generate_csv_export(self, sample_plans, sample_client_brief):
        """Test CSV export generation."""
        component = PlanExportComponent()
        
        options = {
            'include_rationale': True,
            'include_sites': True,
            'include_metrics': True,
            'include_timestamp': True,
            'include_brief': True
        }
        
        with patch('streamlit.download_button') as mock_download, \
             patch('streamlit.success') as mock_success:
            
            result = component._generate_csv_export(
                sample_plans, sample_client_brief, "All Plans", options
            )
            
            assert result['success'] is True
            assert result['format'] == 'CSV'
            assert 'filename' in result
            assert 'content' in result
            assert result['line_count'] > 0
            
            # Check that download button was called
            mock_download.assert_called_once()
            mock_success.assert_called_once()
    
    def test_generate_json_export(self, sample_plans, sample_client_brief):
        """Test JSON export generation."""
        component = PlanExportComponent()
        
        options = {
            'include_rationale': True,
            'include_sites': True,
            'include_metrics': True,
            'include_timestamp': True,
            'include_brief': True
        }
        
        with patch('streamlit.download_button') as mock_download, \
             patch('streamlit.success') as mock_success:
            
            result = component._generate_json_export(
                sample_plans, sample_client_brief, "All Plans", options
            )
            
            assert result['success'] is True
            assert result['format'] == 'JSON'
            assert 'filename' in result
            assert 'content' in result
            assert result['plan_count'] == len(sample_plans)
            
            # Verify JSON content is valid
            import json
            json_data = json.loads(result['content'])
            assert 'export_info' in json_data
            assert 'plans' in json_data
            assert len(json_data['plans']) == len(sample_plans)
            
            # Check that download button was called
            mock_download.assert_called_once()
            mock_success.assert_called_once()
    
    def test_save_plans_to_session(self, sample_plans, sample_client_brief):
        """Test saving plans to session state."""
        component = PlanExportComponent()
        
        # Mock session state properly
        mock_session_state = Mock()
        mock_session_state.saved_plans = []
        
        with patch('streamlit.success') as mock_success, \
             patch('streamlit.session_state', mock_session_state):
            
            result = component._save_plans_to_session(sample_plans, sample_client_brief)
            
            assert result['success'] is True
            assert 'save_id' in result
            assert 'saved_at' in result
            assert result['plan_count'] == len(sample_plans)
            
            # Check that plans were saved to session state
            assert len(mock_session_state.saved_plans) == 1
            
            saved_data = mock_session_state.saved_plans[0]
            assert saved_data['plans'] == sample_plans
            assert saved_data['client_brief'] == sample_client_brief
            
            mock_success.assert_called_once()
    
    def test_prepare_email_export(self, sample_plans, sample_client_brief):
        """Test email export preparation."""
        component = PlanExportComponent()
        
        with patch('streamlit.info') as mock_info, \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.text_area') as mock_text_area:
            
            # Mock the expander context manager
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock()
            
            result = component._prepare_email_export(sample_plans, sample_client_brief)
            
            assert result['success'] is True
            assert 'subject' in result
            assert 'body' in result
            assert result['plan_count'] == len(sample_plans)
            
            # Check email content
            assert sample_client_brief['brand_name'] in result['subject']
            assert sample_client_brief['brand_name'] in result['body']
            
            for plan in sample_plans:
                assert plan.title in result['body']


if __name__ == "__main__":
    pytest.main([__file__])