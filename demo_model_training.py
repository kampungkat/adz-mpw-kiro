#!/usr/bin/env python3
"""
Demo script for Model Training and Fine-tuning capabilities.

This script demonstrates:
1. Collecting training data from successful media plans
2. Exporting data in OpenAI fine-tuning format
3. Managing fine-tuning jobs
4. Comparing model performance
5. Using fine-tuned models in production
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from business_logic.model_training_manager import ModelTrainingManager
from business_logic.ai_plan_generator import AIPlanGenerator
from models.data_models import ClientBrief, MediaPlan, FormatAllocation
from data.manager import DataManager


def create_sample_data():
    """Create sample training data for demonstration."""
    print("Creating sample training data...")
    
    # Sample client briefs and corresponding successful plans
    sample_campaigns = [
        {
            'brief': ClientBrief(
                brand_name="TechCorp",
                budget=75000.0,
                country="US",
                campaign_period="Q1 2024",
                objective="Brand Awareness",
                planning_mode="AI"
            ),
            'plan': MediaPlan(
                plan_id="plan_1",
                title="Multi-Channel Reach Strategy",
                total_budget=74500.0,
                allocations=[
                    FormatAllocation(
                        format_name="Display Banner",
                        budget_allocation=35000.0,
                        cpm=4.50,
                        estimated_impressions=7777778,
                        recommended_sites=["cnn.com", "bbc.com", "reuters.com"],
                        notes="High-visibility news sites for brand awareness"
                    ),
                    FormatAllocation(
                        format_name="Video Pre-roll",
                        budget_allocation=25000.0,
                        cpm=12.00,
                        estimated_impressions=2083333,
                        recommended_sites=["youtube.com", "vimeo.com"],
                        notes="Engaging video content for brand storytelling"
                    ),
                    FormatAllocation(
                        format_name="Native Advertising",
                        budget_allocation=14500.0,
                        cpm=8.00,
                        estimated_impressions=1812500,
                        recommended_sites=["buzzfeed.com", "mashable.com"],
                        notes="Native content for organic engagement"
                    )
                ],
                estimated_reach=5500000,
                estimated_impressions=11673611,
                rationale="Balanced approach focusing on reach across multiple touchpoints. Display provides broad awareness, video delivers engaging brand story, and native ensures organic integration.",
                created_at=datetime.now()
            ),
            'performance': {"ctr": 0.045, "conversion_rate": 0.025, "brand_lift": 0.15}
        },
        {
            'brief': ClientBrief(
                brand_name="FashionBrand",
                budget=50000.0,
                country="US",
                campaign_period="Q2 2024",
                objective="Sales Conversion",
                planning_mode="AI"
            ),
            'plan': MediaPlan(
                plan_id="plan_2",
                title="Conversion-Focused Strategy",
                total_budget=49800.0,
                allocations=[
                    FormatAllocation(
                        format_name="Search Ads",
                        budget_allocation=30000.0,
                        cpm=15.00,
                        estimated_impressions=2000000,
                        recommended_sites=["google.com", "bing.com"],
                        notes="High-intent search targeting for immediate conversions"
                    ),
                    FormatAllocation(
                        format_name="Social Media Ads",
                        budget_allocation=19800.0,
                        cpm=6.50,
                        estimated_impressions=3046154,
                        recommended_sites=["facebook.com", "instagram.com"],
                        notes="Social targeting for fashion-conscious demographics"
                    )
                ],
                estimated_reach=3200000,
                estimated_impressions=5046154,
                rationale="Conversion-optimized strategy prioritizing high-intent channels. Search ads capture immediate purchase intent while social ads build consideration among target demographics.",
                created_at=datetime.now()
            ),
            'performance': {"ctr": 0.065, "conversion_rate": 0.045, "roas": 4.2}
        },
        {
            'brief': ClientBrief(
                brand_name="HealthTech",
                budget=100000.0,
                country="US",
                campaign_period="Q3 2024",
                objective="Lead Generation",
                planning_mode="AI"
            ),
            'plan': MediaPlan(
                plan_id="plan_3",
                title="B2B Lead Generation Strategy",
                total_budget=98500.0,
                allocations=[
                    FormatAllocation(
                        format_name="LinkedIn Sponsored Content",
                        budget_allocation=45000.0,
                        cpm=25.00,
                        estimated_impressions=1800000,
                        recommended_sites=["linkedin.com"],
                        notes="Professional targeting for B2B healthcare decision makers"
                    ),
                    FormatAllocation(
                        format_name="Industry Publication Ads",
                        budget_allocation=35000.0,
                        cpm=18.00,
                        estimated_impressions=1944444,
                        recommended_sites=["healthcareit.com", "modernhealthcare.com"],
                        notes="Industry-specific publications for credibility and reach"
                    ),
                    FormatAllocation(
                        format_name="Webinar Sponsorships",
                        budget_allocation=18500.0,
                        cpm=50.00,
                        estimated_impressions=370000,
                        recommended_sites=["webinar-platforms.com"],
                        notes="Educational content sponsorships for thought leadership"
                    )
                ],
                estimated_reach=2800000,
                estimated_impressions=4114444,
                rationale="B2B focused strategy leveraging professional networks and industry publications. LinkedIn provides precise professional targeting, industry publications offer credibility, and webinar sponsorships establish thought leadership.",
                created_at=datetime.now()
            ),
            'performance': {"ctr": 0.035, "lead_conversion_rate": 0.08, "cost_per_lead": 125.0}
        }
    ]
    
    return sample_campaigns


def demo_training_data_collection():
    """Demonstrate training data collection and validation."""
    print("\n" + "="*60)
    print("DEMO: Training Data Collection")
    print("="*60)
    
    # Initialize training manager
    training_manager = ModelTrainingManager(skip_openai_init=True)
    
    # Create sample data
    sample_campaigns = create_sample_data()
    
    # Collect training data
    for i, campaign in enumerate(sample_campaigns):
        print(f"\nCollecting training data for campaign {i+1}: {campaign['brief'].brand_name}")
        
        success = training_manager.collect_training_data(
            campaign['brief'],
            campaign['plan'],
            performance_metrics=campaign['performance'],
            validated=True
        )
        
        if success:
            print(f"✓ Successfully collected training data for {campaign['brief'].brand_name}")
        else:
            print(f"✗ Failed to collect training data for {campaign['brief'].brand_name}")
    
    # Get training data summary
    summary = training_manager.get_training_data_summary()
    print(f"\nTraining Data Summary:")
    print(f"  Total Examples: {summary['total_examples']}")
    print(f"  Validated Examples: {summary['validated_examples']}")
    print(f"  Ready for Export: {summary['ready_for_export']}")
    
    # Validate requirements
    validation = training_manager.validate_training_data_requirements()
    print(f"\nValidation Results:")
    print(f"  Meets Requirements: {validation['ready_for_fine_tuning']}")
    print(f"  Diversity Score: {validation['diversity_score']:.2f}")
    
    if validation['recommendations']:
        print("  Recommendations:")
        for rec in validation['recommendations']:
            print(f"    - {rec}")
    
    return training_manager


def demo_openai_export(training_manager: ModelTrainingManager):
    """Demonstrate exporting training data in OpenAI format."""
    print("\n" + "="*60)
    print("DEMO: OpenAI Format Export")
    print("="*60)
    
    # Export training data
    success, result = training_manager.export_training_data_for_openai(
        include_unvalidated=False,
        max_examples=50
    )
    
    if success:
        print(f"✓ Successfully exported training data to: {result}")
        
        # Show a sample of the exported data
        try:
            with open(result, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"\nExported {len(lines)} training examples")
                
                if lines:
                    import json
                    sample_example = json.loads(lines[0])
                    print("\nSample training example structure:")
                    print(f"  Messages: {len(sample_example['messages'])}")
                    print(f"  System message length: {len(sample_example['messages'][0]['content'])}")
                    print(f"  User message length: {len(sample_example['messages'][1]['content'])}")
                    print(f"  Assistant message length: {len(sample_example['messages'][2]['content'])}")
        except Exception as e:
            print(f"Could not read exported file: {e}")
    else:
        print(f"✗ Failed to export training data: {result}")
    
    return success


def demo_fine_tuning_job_management(training_manager: ModelTrainingManager):
    """Demonstrate fine-tuning job management (simulation)."""
    print("\n" + "="*60)
    print("DEMO: Fine-tuning Job Management (Simulated)")
    print("="*60)
    
    print("Note: This demo simulates fine-tuning operations without actual OpenAI API calls")
    
    # Simulate job initiation
    print("\n1. Initiating fine-tuning job...")
    print("   Model: gpt-3.5-turbo")
    print("   Training examples: 3")
    print("   Status: Job would be created with OpenAI API")
    
    # Simulate job monitoring
    print("\n2. Monitoring job progress...")
    job_statuses = ["validating_files", "queued", "running", "succeeded"]
    
    for i, status in enumerate(job_statuses):
        print(f"   Step {i+1}: {status}")
        if status == "running":
            print("     Progress: Training in progress...")
            print("     Estimated completion: 15-30 minutes")
        elif status == "succeeded":
            print("     Fine-tuned model: ft:gpt-3.5-turbo:adzymic:media-planner")
    
    # Simulate model deployment
    print("\n3. Deploying fine-tuned model...")
    print("   ✓ Model ft:gpt-3.5-turbo:adzymic:media-planner ready for use")
    
    return "ft:gpt-3.5-turbo:adzymic:media-planner"


def demo_model_performance_comparison():
    """Demonstrate model performance comparison and selection."""
    print("\n" + "="*60)
    print("DEMO: Model Performance Comparison")
    print("="*60)
    
    # Initialize AI plan generator with performance tracking
    ai_generator = AIPlanGenerator(skip_openai_init=True)
    
    # Simulate performance data
    print("Simulating model performance data...")
    
    # Base model performance
    for i in range(10):
        ai_generator.track_model_performance(
            model_used="gpt-4",
            response_time=2.5 + (i * 0.1),
            success=True,
            quality_score=0.75 + (i * 0.02)
        )
    
    # Fine-tuned model performance
    for i in range(10):
        ai_generator.track_model_performance(
            model_used="ft:gpt-3.5-turbo:adzymic:media-planner",
            response_time=1.8 + (i * 0.1),
            success=True,
            quality_score=0.82 + (i * 0.015)
        )
    
    # Get performance summary
    model_info = ai_generator.get_model_info()
    performance_summary = model_info['performance_summary']
    
    print("\nPerformance Comparison:")
    print(f"Base Model (gpt-4):")
    print(f"  Avg Response Time: {performance_summary['base_model']['avg_response_time']:.2f}s")
    print(f"  Avg Quality Score: {performance_summary['base_model']['avg_quality_score']:.3f}")
    print(f"  Success Rate: {performance_summary['base_model']['success_rate']:.1%}")
    
    print(f"\nFine-tuned Model:")
    print(f"  Avg Response Time: {performance_summary['fine_tuned_model']['avg_response_time']:.2f}s")
    print(f"  Avg Quality Score: {performance_summary['fine_tuned_model']['avg_quality_score']:.3f}")
    print(f"  Success Rate: {performance_summary['fine_tuned_model']['success_rate']:.1%}")
    
    # Test model selection
    ai_generator.fine_tuned_model = "ft:gpt-3.5-turbo:adzymic:media-planner"
    ai_generator.set_model_selection_strategy("auto")
    
    sample_brief = ClientBrief(
        brand_name="TestBrand",
        budget=50000.0,
        country="US",
        campaign_period="Q4 2024",
        objective="Brand Awareness",
        planning_mode="AI"
    )
    
    selected_model = ai_generator.select_optimal_model(sample_brief)
    print(f"\nAuto-selected model for new campaign: {selected_model}")
    
    # Cost analysis
    print("\nSimulating cost tracking...")
    ai_generator.track_model_cost("gpt-4", 1500, 800)
    ai_generator.track_model_cost("ft:gpt-3.5-turbo:adzymic:media-planner", 1500, 800)
    
    cost_analysis = ai_generator.get_cost_analysis()
    print(f"Cost Analysis:")
    print(f"  Total Cost: ${cost_analysis['total_cost']:.4f}")
    print(f"  Base Model Cost/Call: ${cost_analysis['base_model']['cost_per_call']:.4f}")
    print(f"  Fine-tuned Cost/Call: ${cost_analysis['fine_tuned_model']['cost_per_call']:.4f}")
    
    if cost_analysis['recommendations']:
        print("  Recommendations:")
        for rec in cost_analysis['recommendations']:
            print(f"    - {rec}")


def demo_plan_quality_evaluation():
    """Demonstrate plan quality evaluation."""
    print("\n" + "="*60)
    print("DEMO: Plan Quality Evaluation")
    print("="*60)
    
    ai_generator = AIPlanGenerator(skip_openai_init=True)
    
    # Create sample plans with different quality levels
    sample_campaigns = create_sample_data()
    
    print("Evaluating plan quality for sample campaigns:")
    
    for i, campaign in enumerate(sample_campaigns):
        quality_score = ai_generator.evaluate_plan_quality(
            campaign['plan'], 
            campaign['brief']
        )
        
        print(f"\nCampaign {i+1}: {campaign['brief'].brand_name}")
        print(f"  Plan: {campaign['plan'].title}")
        print(f"  Quality Score: {quality_score:.3f}")
        print(f"  Budget Adherence: {campaign['plan'].total_budget / campaign['brief'].budget:.1%}")
        print(f"  Format Diversity: {len(campaign['plan'].allocations)} formats")
        print(f"  Rationale Length: {len(campaign['plan'].rationale)} characters")


def main():
    """Run the complete model training demo."""
    print("AI Media Planner - Model Training & Fine-tuning Demo")
    print("="*60)
    
    try:
        # Demo 1: Training data collection
        training_manager = demo_training_data_collection()
        
        # Demo 2: OpenAI format export
        export_success = demo_openai_export(training_manager)
        
        # Demo 3: Fine-tuning job management (simulated)
        if export_success:
            fine_tuned_model = demo_fine_tuning_job_management(training_manager)
        
        # Demo 4: Model performance comparison
        demo_model_performance_comparison()
        
        # Demo 5: Plan quality evaluation
        demo_plan_quality_evaluation()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Training data collection and validation")
        print("✓ OpenAI fine-tuning format export")
        print("✓ Fine-tuning job management workflow")
        print("✓ Model performance tracking and comparison")
        print("✓ Automatic model selection based on performance")
        print("✓ Cost tracking and optimization")
        print("✓ Plan quality evaluation and scoring")
        
        print("\nNext Steps:")
        print("1. Collect real training data from successful campaigns")
        print("2. Export data and initiate actual fine-tuning with OpenAI")
        print("3. Monitor training progress and deploy fine-tuned model")
        print("4. Compare performance and optimize model selection")
        print("5. Continuously improve with new training data")
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()