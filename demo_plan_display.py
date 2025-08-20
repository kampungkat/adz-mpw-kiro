"""
Demo script showing the PlanDisplayComponent and PlanExportComponent in action.

This script demonstrates how to use the new plan display and export functionality
with sample data.
"""

import streamlit as st
from datetime import datetime
from ui.components import PlanDisplayComponent, PlanExportComponent
from models.data_models import MediaPlan, FormatAllocation


def create_sample_data():
    """Create sample media plans and client brief for demonstration."""
    
    # Sample allocations for Plan 1
    allocation1_1 = FormatAllocation(
        format_name="Display Banner",
        budget_allocation=15000.0,
        cpm=5.50,
        estimated_impressions=2727272,
        recommended_sites=["Premium News Site", "Tech Blog Network", "Lifestyle Magazine"],
        notes="High-visibility placements for maximum brand exposure"
    )
    
    allocation1_2 = FormatAllocation(
        format_name="Video Pre-roll",
        budget_allocation=10000.0,
        cpm=8.00,
        estimated_impressions=1250000,
        recommended_sites=["YouTube Premium", "Streaming Platform A"],
        notes="Premium video inventory for engagement"
    )
    
    # Plan 1: Reach-Focused Strategy
    plan1 = MediaPlan(
        plan_id="plan_1_demo",
        title="Reach-Focused Strategy",
        total_budget=25000.0,
        allocations=[allocation1_1, allocation1_2],
        estimated_reach=1500000,
        estimated_impressions=3977272,
        rationale="This plan maximizes reach through cost-effective display placements combined with high-impact video advertising. The strategy focuses on broad audience exposure across premium inventory to build brand awareness efficiently.",
        created_at=datetime.now()
    )
    
    # Sample allocations for Plan 2
    allocation2_1 = FormatAllocation(
        format_name="Native Advertising",
        budget_allocation=18000.0,
        cpm=12.00,
        estimated_impressions=1500000,
        recommended_sites=["Premium Publisher 1", "Quality Content Site"],
        notes="High-quality native placements for brand safety"
    )
    
    allocation2_2 = FormatAllocation(
        format_name="Social Media Sponsored",
        budget_allocation=7000.0,
        cpm=6.50,
        estimated_impressions=1076923,
        recommended_sites=["Facebook", "Instagram", "LinkedIn"],
        notes="Targeted social media engagement"
    )
    
    # Plan 2: Quality-Focused Strategy
    plan2 = MediaPlan(
        plan_id="plan_2_demo",
        title="Quality-Focused Strategy",
        total_budget=25000.0,
        allocations=[allocation2_1, allocation2_2],
        estimated_reach=900000,
        estimated_impressions=2576923,
        rationale="This plan prioritizes high-quality placements and brand safety through premium native advertising and targeted social media campaigns. The approach focuses on engagement quality over quantity.",
        created_at=datetime.now()
    )
    
    # Sample allocations for Plan 3
    allocation3_1 = FormatAllocation(
        format_name="Connected TV",
        budget_allocation=12000.0,
        cpm=15.00,
        estimated_impressions=800000,
        recommended_sites=["Streaming Service A", "Smart TV Platform"],
        notes="Premium CTV inventory for high-impact delivery"
    )
    
    allocation3_2 = FormatAllocation(
        format_name="Display Banner",
        budget_allocation=8000.0,
        cpm=5.00,
        estimated_impressions=1600000,
        recommended_sites=["News Network", "Entertainment Sites"],
        notes="Complementary display support"
    )
    
    allocation3_3 = FormatAllocation(
        format_name="Audio Streaming",
        budget_allocation=5000.0,
        cpm=10.00,
        estimated_impressions=500000,
        recommended_sites=["Spotify", "Podcast Network"],
        notes="Audio engagement for commuter audience"
    )
    
    # Plan 3: Multi-Channel Strategy
    plan3 = MediaPlan(
        plan_id="plan_3_demo",
        title="Multi-Channel Strategy",
        total_budget=25000.0,
        allocations=[allocation3_1, allocation3_2, allocation3_3],
        estimated_reach=1200000,
        estimated_impressions=2900000,
        rationale="This plan leverages multiple channels including Connected TV, display, and audio to create a comprehensive media presence. The diversified approach ensures broad reach across different consumption moments.",
        created_at=datetime.now()
    )
    
    # Sample client brief
    client_brief = {
        'brand_name': 'TechStart Pro',
        'budget': 30000.0,
        'country': 'SG',
        'objective': 'Brand Awareness',
        'start_date': '2024-02-01',
        'end_date': '2024-02-29',
        'planning_mode': 'AI Selection',
        'notes': 'Focus on tech-savvy professionals aged 25-45'
    }
    
    return [plan1, plan2, plan3], client_brief


def main():
    """Main demo application."""
    st.set_page_config(
        page_title="Media Plan Display Demo",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Media Plan Display & Export Demo")
    st.markdown("This demo showcases the new plan display and export functionality.")
    
    # Create sample data
    plans, client_brief = create_sample_data()
    
    # Initialize components
    display_component = PlanDisplayComponent()
    export_component = PlanExportComponent()
    
    # Sidebar for demo controls
    st.sidebar.header("Demo Controls")
    
    demo_mode = st.sidebar.selectbox(
        "Select Demo Mode:",
        ["Plan Comparison", "Interactive Analysis", "Export Options", "All Features"]
    )
    
    if demo_mode == "Plan Comparison" or demo_mode == "All Features":
        st.header("📋 Plan Comparison View")
        
        # Render plan comparison
        comparison_result = display_component.render_plan_comparison(plans, client_brief)
        
        if comparison_result.get('selected_plan_index') is not None:
            st.success(f"Selected plan: {plans[comparison_result['selected_plan_index']].title}")
    
    if demo_mode == "Interactive Analysis" or demo_mode == "All Features":
        st.header("📊 Interactive Budget Analysis")
        
        # Render interactive budget breakdown
        analysis_result = display_component.render_interactive_budget_breakdown(plans, client_brief)
        
        if analysis_result.get('selected_plan'):
            st.info(f"Analyzing: {analysis_result['selected_plan'].title}")
    
    if demo_mode == "Export Options" or demo_mode == "All Features":
        st.header("📥 Export Functionality")
        
        # Render export interface
        export_result = export_component.render_export_interface(
            plans, client_brief, selected_plan_index=0
        )
        
        if export_result:
            st.success("Export functionality demonstrated!")
    
    # Demo information
    with st.expander("ℹ️ About This Demo"):
        st.write("""
        **This demo showcases:**
        
        1. **Plan Comparison**: Side-by-side comparison of multiple media plans
        2. **Interactive Analysis**: Detailed budget breakdowns with charts and metrics
        3. **Export Options**: CSV, JSON, and report generation capabilities
        4. **Optimization Insights**: AI-powered recommendations and analysis
        
        **Sample Data Includes:**
        - 3 different strategic approaches (Reach, Quality, Multi-Channel)
        - Realistic budget allocations and CPM rates
        - Performance projections and site recommendations
        - Strategic rationales for each plan
        
        **Key Features:**
        - Interactive charts and visualizations
        - Comprehensive export formats
        - Plan scoring and optimization recommendations
        - Budget utilization analysis
        - Format mix insights
        """)
    
    # Technical details
    with st.expander("🔧 Technical Implementation"):
        st.write("""
        **Components Implemented:**
        
        1. **PlanDisplayComponent**:
           - `render_plan_comparison()`: Side-by-side plan comparison
           - `render_interactive_budget_breakdown()`: Interactive analysis with charts
           - Performance scoring and optimization recommendations
           
        2. **PlanExportComponent**:
           - `render_export_interface()`: Complete export functionality
           - CSV, JSON, and text report generation
           - Session state persistence
           - Email and sharing preparation
        
        **Key Technologies:**
        - Streamlit for interactive UI
        - Pandas for data manipulation
        - Plotly for advanced charts (optional)
        - JSON for data serialization
        - Session state for persistence
        """)


if __name__ == "__main__":
    main()