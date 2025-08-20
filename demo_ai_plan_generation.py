#!/usr/bin/env python3
"""
Demo script for AI Plan Generation System.

This script demonstrates the complete AI plan generation workflow
including budget optimization, plan validation, and system integration.
"""

import json
from datetime import datetime
from models.data_models import ClientBrief
from business_logic.budget_optimizer import BudgetOptimizer, OptimizationStrategy
from business_logic.plan_validator import PlanValidator
from business_logic.media_plan_controller import MediaPlanController
from data.manager import DataManager


def demo_budget_optimizer():
    """Demonstrate budget optimization capabilities."""
    print("=" * 60)
    print("BUDGET OPTIMIZER DEMO")
    print("=" * 60)
    
    # Create sample client brief
    client_brief = ClientBrief(
        brand_name="Demo Brand",
        budget=15000.0,
        country="SG",
        campaign_period="Q1 2024",
        objective="Brand Awareness",
        planning_mode="AI"
    )
    
    # Sample available formats
    available_formats = {
        'Display Banner': {
            'cpm': 2.50,
            'sites': ['site1.com', 'site2.com', 'site3.com']
        },
        'Video Pre-roll': {
            'cpm': 8.00,
            'sites': ['video1.com', 'video2.com']
        },
        'Native Content': {
            'cpm': 4.00,
            'sites': ['native1.com', 'native2.com', 'native3.com']
        },
        'Mobile Banner': {
            'cpm': 3.00,
            'sites': ['mobile1.com', 'mobile2.com']
        }
    }
    
    optimizer = BudgetOptimizer()
    
    # Test different optimization strategies
    strategies = [
        OptimizationStrategy.REACH_FOCUSED,
        OptimizationStrategy.FREQUENCY_FOCUSED,
        OptimizationStrategy.BALANCED,
        OptimizationStrategy.COST_EFFICIENT
    ]
    
    for strategy in strategies:
        print(f"\n{strategy.value.upper()} STRATEGY:")
        print("-" * 40)
        
        result = optimizer.optimize_budget_allocation(
            client_brief, available_formats, strategy
        )
        
        print(f"Total Budget Used: ${result.total_budget_used:,.2f}")
        print(f"Estimated Reach: {result.estimated_total_reach:,}")
        print(f"Estimated Impressions: {result.estimated_total_impressions:,}")
        print(f"Average Frequency: {result.average_frequency:.2f}x")
        print(f"Optimization Score: {result.optimization_score:.1f}/100")
        print(f"Strategy Notes: {result.strategy_notes}")
        
        print("\nAllocations:")
        for allocation in result.allocations:
            print(f"  • {allocation.format_name}: ${allocation.budget_allocation:,.2f} "
                  f"(CPM: ${allocation.cpm:.2f}, Impressions: {allocation.estimated_impressions:,})")


def demo_plan_validator():
    """Demonstrate plan validation capabilities."""
    print("\n" + "=" * 60)
    print("PLAN VALIDATOR DEMO")
    print("=" * 60)
    
    # Sample client brief
    client_brief = ClientBrief(
        brand_name="Demo Brand",
        budget=10000.0,
        country="SG",
        campaign_period="Q1 2024",
        objective="Brand Awareness",
        planning_mode="AI"
    )
    
    # Sample available formats
    available_formats = {
        'impact_formats': {
            'Display Banner': {'cpm': 2.50},
            'Video Pre-roll': {'cpm': 8.00}
        },
        'reach_formats': {
            'Native Content': {'cpm': 4.00}
        }
    }
    
    validator = PlanValidator()
    
    # Test valid plans
    print("\nTesting VALID plans:")
    print("-" * 30)
    
    valid_plans = {
        'plans': [
            {
                'title': 'Reach-Focused Strategy',
                'rationale': 'Maximize unique audience reach across multiple touchpoints',
                'total_budget': 9500.0,
                'estimated_reach': 45000,
                'estimated_impressions': 2500000,
                'allocations': [
                    {
                        'format_name': 'Display Banner',
                        'budget_allocation': 6000.0,
                        'cpm': 2.50,
                        'estimated_impressions': 2400000,
                        'recommended_sites': ['site1.com', 'site2.com'],
                        'notes': 'Primary reach driver with cost efficiency'
                    },
                    {
                        'format_name': 'Native Content',
                        'budget_allocation': 3500.0,
                        'cpm': 4.00,
                        'estimated_impressions': 875000,
                        'recommended_sites': ['native1.com'],
                        'notes': 'Quality engagement complement'
                    }
                ]
            },
            {
                'title': 'Impact-Focused Strategy',
                'rationale': 'Prioritize high-impact video content for brand memorability',
                'total_budget': 9200.0,
                'estimated_reach': 35000,
                'estimated_impressions': 1800000,
                'allocations': [
                    {
                        'format_name': 'Video Pre-roll',
                        'budget_allocation': 6400.0,
                        'cpm': 8.00,
                        'estimated_impressions': 800000,
                        'recommended_sites': ['video1.com'],
                        'notes': 'Premium video placement for maximum impact'
                    },
                    {
                        'format_name': 'Display Banner',
                        'budget_allocation': 2800.0,
                        'cpm': 2.50,
                        'estimated_impressions': 1120000,
                        'recommended_sites': ['site1.com'],
                        'notes': 'Supporting reach extension'
                    }
                ]
            }
        ]
    }
    
    result = validator.parse_and_validate_plans(
        valid_plans, client_brief, available_formats
    )
    
    print(f"Validation Result: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"Valid Plans: {len(result.parsed_plans)}")
    print(f"Errors: {result.total_errors}")
    print(f"Warnings: {result.total_warnings}")
    
    if result.issues:
        print("\nIssues found:")
        for issue in result.issues:
            print(f"  • {issue.severity.value.upper()}: {issue.message}")
    
    # Test invalid plans
    print("\nTesting INVALID plans:")
    print("-" * 30)
    
    invalid_plans = {
        'plans': [
            {
                'title': 'Over-Budget Plan',
                'rationale': 'This plan exceeds the budget',
                'total_budget': 15000.0,  # Exceeds $10k budget
                'allocations': [
                    {
                        'format_name': 'Video Pre-roll',
                        'budget_allocation': 15000.0,
                        'cpm': 8.00,
                        'estimated_impressions': 1875000
                    }
                ]
            }
        ]
    }
    
    result = validator.parse_and_validate_plans(
        invalid_plans, client_brief, available_formats
    )
    
    print(f"Validation Result: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"Valid Plans: {len(result.parsed_plans)}")
    print(f"Errors: {result.total_errors}")
    print(f"Warnings: {result.total_warnings}")
    
    if result.issues:
        print("\nIssues found:")
        for issue in result.issues:
            print(f"  • {issue.severity.value.upper()}: {issue.message}")


def demo_system_integration():
    """Demonstrate system integration capabilities."""
    print("\n" + "=" * 60)
    print("SYSTEM INTEGRATION DEMO")
    print("=" * 60)
    
    # Create controller in testing mode
    controller = MediaPlanController(testing_mode=True)
    
    # Sample client brief
    client_brief = ClientBrief(
        brand_name="Integration Test Brand",
        budget=12000.0,
        country="SG",
        campaign_period="Q1 2024",
        objective="Brand Awareness",
        planning_mode="AI"
    )
    
    print("Testing input validation...")
    print("-" * 30)
    
    # Test input validation
    is_valid, message = controller.validate_inputs(client_brief)
    print(f"Input Validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"Message: {message}")
    
    # Test with invalid inputs
    invalid_brief = ClientBrief(
        brand_name="",  # Empty brand name
        budget=-1000.0,  # Negative budget
        country="XX",  # Invalid country
        campaign_period="Q1 2024",
        objective="Brand Awareness",
        planning_mode="AI"
    )
    
    is_valid, message = controller.validate_inputs(invalid_brief)
    print(f"\nInvalid Input Test: {'PASSED' if not is_valid else 'FAILED'}")
    print(f"Error Message: {message}")
    
    print("\nTesting system status...")
    print("-" * 30)
    
    # Test system status
    try:
        status = controller.get_system_status()
        print(f"System Ready: {status.get('system_ready', False)}")
        print(f"Available Markets: {status.get('market_count', 0)}")
        print(f"Data Status: {status.get('data_status', {}).get('overall_status', 'unknown')}")
    except Exception as e:
        print(f"System status check failed (expected in demo): {str(e)}")


def main():
    """Run all demos."""
    print("AI PLAN GENERATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the AI plan generation system components:")
    print("• Budget Optimizer: Optimizes budget allocation across formats")
    print("• Plan Validator: Validates AI-generated plans for correctness")
    print("• System Integration: Coordinates all components")
    print()
    
    try:
        demo_budget_optimizer()
        demo_plan_validator()
        demo_system_integration()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("All components are working correctly.")
        print("The AI plan generation system is ready for integration.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()