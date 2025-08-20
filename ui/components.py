"""
UI components for the AI Media Planner application.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import logging
import pandas as pd

from data.manager import DataManager

logger = logging.getLogger(__name__)


class MediaPlannerForm:
    """
    MediaPlannerForm component for collecting client information and planning preferences.
    
    Handles form fields for brand name, budget, country, campaign period, objective,
    and planning mode with real-time validation and dynamic country selection.
    """
    
    def __init__(self, data_manager: DataManager):
        """
        Initialize the MediaPlannerForm.
        
        Args:
            data_manager: DataManager instance for accessing market data
        """
        self.data_manager = data_manager
        self._form_data = {}
        self._validation_errors = {}
    
    def render(self) -> Tuple[Dict[str, Any], bool]:
        """
        Render the media planner form and return form data and validation status.
        
        Returns:
            Tuple of (form_data, is_valid)
        """
        st.subheader("📋 Campaign Information")
        
        # Initialize form data
        form_data = {}
        is_valid = True
        
        # Create form container
        with st.form("media_planner_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Brand/Advertiser Name
                brand_name = st.text_input(
                    "Brand/Advertiser Name *",
                    value=st.session_state.get('brand_name', ''),
                    placeholder="Enter brand or advertiser name",
                    help="The name of the brand or advertiser for this campaign"
                )
                form_data['brand_name'] = brand_name.strip()
                
                # Budget
                budget = st.number_input(
                    "Campaign Budget (USD) *",
                    min_value=0.0,
                    value=st.session_state.get('budget', 0.0),
                    step=1000.0,
                    format="%.2f",
                    help="Total budget available for this campaign"
                )
                form_data['budget'] = budget
                
                # Campaign Period Start
                start_date = st.date_input(
                    "Campaign Start Date *",
                    value=st.session_state.get('start_date', date.today()),
                    help="When the campaign should begin"
                )
                form_data['start_date'] = start_date
            
            with col2:
                # Country Selection with dynamic loading
                available_markets = self._get_available_markets()
                
                if available_markets:
                    country_options = ['Select a country...'] + available_markets
                    country_index = 0
                    
                    # Preserve previous selection if valid
                    if 'country' in st.session_state and st.session_state['country'] in available_markets:
                        country_index = country_options.index(st.session_state['country'])
                    
                    country = st.selectbox(
                        "Target Country *",
                        options=country_options,
                        index=country_index,
                        help="Select the target market for this campaign"
                    )
                    
                    if country == 'Select a country...':
                        country = ''
                    form_data['country'] = country
                else:
                    st.error("❌ No markets available. Please check data files.")
                    form_data['country'] = ''
                
                # Campaign Period End
                end_date = st.date_input(
                    "Campaign End Date *",
                    value=st.session_state.get('end_date', date.today()),
                    help="When the campaign should end"
                )
                form_data['end_date'] = end_date
                
                # Campaign Objective
                objective_options = [
                    'Brand Awareness',
                    'Reach & Frequency',
                    'Traffic Generation',
                    'Lead Generation',
                    'Conversions',
                    'Engagement',
                    'Video Views',
                    'App Installs'
                ]
                
                objective = st.selectbox(
                    "Campaign Objective *",
                    options=['Select objective...'] + objective_options,
                    index=0 if 'objective' not in st.session_state else 
                           (objective_options.index(st.session_state['objective']) + 1 
                            if st.session_state['objective'] in objective_options else 0),
                    help="Primary goal for this campaign"
                )
                
                if objective == 'Select objective...':
                    objective = ''
                form_data['objective'] = objective
            
            # Planning Mode Toggle
            st.subheader("🎯 Planning Mode")
            
            planning_mode = st.radio(
                "How would you like to select ad products?",
                options=['AI Selection', 'Manual Selection'],
                index=0 if st.session_state.get('planning_mode', 'AI Selection') == 'AI Selection' else 1,
                help="Choose whether to let AI recommend products or manually select them",
                horizontal=True
            )
            form_data['planning_mode'] = planning_mode
            
            # Additional Notes (Optional)
            notes = st.text_area(
                "Additional Notes (Optional)",
                value=st.session_state.get('notes', ''),
                placeholder="Any additional requirements or preferences for this campaign...",
                help="Optional field for any special requirements or notes"
            )
            form_data['notes'] = notes.strip()
            
            # Form submission
            submitted = st.form_submit_button("Generate Media Plans", type="primary")
            
            if submitted:
                # Validate form data
                validation_errors = self._validate_form_data(form_data)
                
                if validation_errors:
                    is_valid = False
                    self._display_validation_errors(validation_errors)
                else:
                    # Store in session state for persistence
                    for key, value in form_data.items():
                        st.session_state[key] = value
                    
                    st.success("✅ Form validated successfully!")
                    is_valid = True
        
        # Display real-time budget validation outside the form
        if form_data.get('budget', 0) > 0:
            self._display_budget_validation(form_data['budget'], form_data.get('country', ''))
        
        return form_data, is_valid and submitted
    
    def _get_available_markets(self) -> List[str]:
        """
        Get list of available markets from data manager with comprehensive error handling.
        
        Returns:
            List of market codes
        """
        try:
            markets = self.data_manager.get_available_markets()
            if not markets:
                logger.warning("No markets found in data sources")
                self._display_data_loading_error("No markets available", 
                    "Check that rate card and site list files are properly formatted and contain market data")
            return markets
        except FileNotFoundError as e:
            logger.error(f"Data files not found: {str(e)}")
            self._display_data_loading_error("Data Files Missing", 
                "Required Excel files (rate card and site list) are not found in the application directory")
            return []
        except PermissionError as e:
            logger.error(f"Permission error accessing data files: {str(e)}")
            self._display_data_loading_error("File Access Error", 
                "Cannot access data files. Check file permissions")
            return []
        except Exception as e:
            logger.error(f"Error loading available markets: {str(e)}")
            self._display_data_loading_error("Data Loading Error", 
                f"Unexpected error loading market data: {str(e)}")
            return []
    
    def _display_data_loading_error(self, title: str, message: str):
        """
        Display user-friendly error messages for data loading issues.
        
        Args:
            title: Error title
            message: Detailed error message
        """
        st.error(f"❌ **{title}**")
        st.error(message)
        
        with st.expander("🔧 Troubleshooting Steps"):
            st.write("**To resolve data loading issues:**")
            st.write("1. **Check File Presence**: Ensure these files exist in the application directory:")
            st.write("   • `adzymic_rate_card.xlsx` - Contains CPM rates for different markets")
            st.write("   • `APX Sitelist - Regional.xlsx` - Contains site categorization by market")
            st.write("")
            st.write("2. **Verify File Format**: Files should be Excel format (.xlsx or .xls)")
            st.write("")
            st.write("3. **Check File Content**: Files should contain:")
            st.write("   • Rate card: Market codes as columns, format names as rows")
            st.write("   • Site list: Market-specific sheets with site categorization")
            st.write("")
            st.write("4. **File Permissions**: Ensure the application can read the files")
            st.write("")
            st.write("5. **Contact Support**: If issues persist, contact your system administrator")
    
    def _validate_form_data(self, form_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate form data and return validation errors.
        
        Args:
            form_data: Form data to validate
            
        Returns:
            Dictionary of field names to error messages
        """
        errors = {}
        
        # Required field validation
        required_fields = {
            'brand_name': 'Brand/Advertiser Name',
            'budget': 'Campaign Budget',
            'country': 'Target Country',
            'start_date': 'Campaign Start Date',
            'end_date': 'Campaign End Date',
            'objective': 'Campaign Objective'
        }
        
        for field, display_name in required_fields.items():
            value = form_data.get(field)
            if not value or (isinstance(value, str) and not value.strip()):
                errors[field] = f"{display_name} is required"
        
        # Brand name validation
        brand_name = form_data.get('brand_name', '').strip()
        if brand_name:
            if len(brand_name) < 2:
                errors['brand_name'] = "Brand name must be at least 2 characters long"
            elif len(brand_name) > 100:
                errors['brand_name'] = "Brand name must be less than 100 characters"
            elif not brand_name.replace(' ', '').replace('-', '').replace('_', '').isalnum():
                errors['brand_name'] = "Brand name contains invalid characters"
        
        # Budget validation with detailed ranges
        budget = form_data.get('budget', 0)
        if budget <= 0:
            errors['budget'] = "Budget must be greater than 0"
        elif budget < 1000:
            errors['budget'] = "Budget should be at least $1,000 for effective media planning"
        elif budget > 10000000:  # 10 million
            errors['budget'] = "Budget exceeds maximum limit of $10,000,000"
        
        # Enhanced budget range suggestions
        country = form_data.get('country', '')
        if budget > 0 and country:
            budget_suggestions = self._get_budget_suggestions(budget, country)
            if budget_suggestions.get('warning'):
                errors['budget'] = budget_suggestions['warning']
        
        # Date validation with business logic
        start_date = form_data.get('start_date')
        end_date = form_data.get('end_date')
        
        if start_date and end_date:
            if end_date <= start_date:
                errors['end_date'] = "End date must be after start date"
            
            # Check if start date is too far in the past
            from datetime import date, timedelta
            today = date.today()
            if start_date < today - timedelta(days=30):
                errors['start_date'] = "Start date cannot be more than 30 days in the past"
            
            # Check if start date is too far in the future
            if start_date > today + timedelta(days=365):
                errors['start_date'] = "Start date cannot be more than 1 year in the future"
            
            # Check campaign period
            campaign_days = (end_date - start_date).days
            if campaign_days < 7:
                errors['end_date'] = "Campaign period should be at least 7 days for meaningful results"
            elif campaign_days > 365:
                errors['end_date'] = "Campaign period should not exceed 365 days"
            elif campaign_days < 14:
                # Warning for short campaigns
                errors['end_date'] = "Campaigns shorter than 14 days may have limited optimization opportunities"
        
        # Country validation with market data check
        if country:
            available_markets = self._get_available_markets()
            if not available_markets:
                errors['country'] = "No market data available. Please check data files."
            elif country not in available_markets:
                errors['country'] = f"Selected country '{country}' is not available. Available markets: {', '.join(available_markets[:5])}{'...' if len(available_markets) > 5 else ''}"
            else:
                # Validate market data completeness
                try:
                    market_data = self.data_manager.get_market_data(country)
                    if not market_data.get('available'):
                        errors['country'] = f"Incomplete data for {country}. Some features may be limited."
                    else:
                        # Check if market has sufficient format options
                        total_formats = (
                            len(market_data.get('rate_card', {}).get('impact_formats', {})) +
                            len(market_data.get('rate_card', {}).get('reach_formats', {}))
                        )
                        if total_formats < 2:
                            errors['country'] = f"Limited ad format options available for {country} (only {total_formats} formats)"
                except Exception as e:
                    logger.error(f"Error validating market data: {str(e)}")
                    errors['country'] = f"Error validating market data for {country}"
        
        return errors
    
    def _get_budget_suggestions(self, budget: float, country: str) -> Dict[str, str]:
        """
        Get budget-specific suggestions and warnings for a market.
        
        Args:
            budget: Campaign budget
            country: Target country
            
        Returns:
            Dictionary with suggestions and warnings
        """
        suggestions = {}
        
        try:
            market_data = self.data_manager.get_market_data(country)
            
            if market_data.get('available'):
                # Get average rates for the market
                impact_rates = list(market_data.get('rate_card', {}).get('impact_formats', {}).values())
                reach_rates = list(market_data.get('rate_card', {}).get('reach_formats', {}).values())
                
                all_rates = impact_rates + reach_rates
                if all_rates:
                    avg_rate = sum(all_rates) / len(all_rates)
                    min_rate = min(all_rates)
                    max_rate = max(all_rates)
                    
                    # Calculate minimum viable budget for meaningful reach
                    min_impressions_needed = 100000  # 100K impressions minimum
                    min_budget_needed = (min_impressions_needed * min_rate) / 1000
                    
                    if budget < min_budget_needed:
                        suggestions['warning'] = (
                            f"Budget may be insufficient for {country}. "
                            f"Minimum recommended: ${min_budget_needed:,.0f} "
                            f"(based on {min_impressions_needed:,} impressions at lowest rate ${min_rate:.2f} CPM)"
                        )
                    
                    # Suggest optimal budget ranges
                    optimal_min = min_budget_needed * 2
                    optimal_max = min_budget_needed * 10
                    
                    if budget >= optimal_min and budget <= optimal_max:
                        suggestions['info'] = f"Budget is in optimal range for {country} (${optimal_min:,.0f} - ${optimal_max:,.0f})"
                    elif budget > optimal_max:
                        suggestions['info'] = f"High budget allows for premium placements and comprehensive reach in {country}"
        
        except Exception as e:
            logger.error(f"Error getting budget suggestions: {str(e)}")
        
        return suggestions


class PlanDisplayComponent:
    """
    Component for displaying and comparing generated media plans.
    
    Provides side-by-side plan comparison, detailed breakdowns, interactive elements,
    and comprehensive plan information display.
    """
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """
        Initialize the PlanDisplayComponent.
        
        Args:
            data_manager: DataManager instance for accessing market data
        """
        self.data_manager = data_manager
    
    def render_plan_comparison(self, plans: List[Any], client_brief: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render side-by-side comparison of multiple media plans.
        
        Args:
            plans: List of MediaPlan objects to display
            client_brief: Client brief information for context
            
        Returns:
            Dictionary with user selections and interactions
        """
        if not plans:
            st.warning("No media plans available to display.")
            return {}
        
        st.subheader("📊 Generated Media Plans")
        
        # Plan selection for detailed view
        selected_plan_index = None
        
        # Display plans in columns for comparison
        if len(plans) == 1:
            cols = [st.container()]
        elif len(plans) == 2:
            cols = st.columns(2)
        else:
            cols = st.columns(3)
        
        plan_interactions = {}
        
        for i, plan in enumerate(plans[:3]):  # Limit to 3 plans for display
            with cols[i % len(cols)]:
                # Plan header with selection
                plan_selected = st.button(
                    f"📋 {plan.title}",
                    key=f"select_plan_{i}",
                    help=f"Click to view detailed breakdown of {plan.title}",
                    use_container_width=True
                )
                
                if plan_selected:
                    selected_plan_index = i
                
                # Plan summary card
                with st.container():
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 8px 0;">
                        <h4 style="margin-top: 0; color: #1f77b4;">{plan.title}</h4>
                        <p><strong>Total Budget:</strong> ${plan.total_budget:,.2f}</p>
                        <p><strong>Estimated Impressions:</strong> {plan.estimated_impressions:,}</p>
                        <p><strong>Estimated Reach:</strong> {plan.estimated_reach:,}</p>
                        <p><strong>Formats Used:</strong> {len(plan.allocations)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Quick allocation preview
                st.write("**Budget Allocation:**")
                for allocation in plan.allocations:
                    percentage = (allocation.budget_allocation / plan.total_budget * 100) if plan.total_budget > 0 else 0
                    st.write(f"• {allocation.format_name}: ${allocation.budget_allocation:,.0f} ({percentage:.1f}%)")
                
                # Plan rationale (truncated)
                with st.expander("Strategy Rationale"):
                    st.write(plan.rationale)
                
                # Export button for individual plan
                plan_interactions[f"export_plan_{i}"] = st.button(
                    "📥 Export Plan",
                    key=f"export_{i}",
                    help=f"Export {plan.title} to CSV"
                )
        
        # Detailed view for selected plan
        if selected_plan_index is not None:
            self._render_detailed_plan_view(plans[selected_plan_index], client_brief)
        
        # Comparison summary
        if len(plans) > 1:
            self._render_comparison_summary(plans, client_brief)
        
        return {
            'selected_plan_index': selected_plan_index,
            'plan_interactions': plan_interactions
        }
    
    def _render_detailed_plan_view(self, plan: Any, client_brief: Dict[str, Any]):
        """
        Render detailed view of a single media plan.
        
        Args:
            plan: MediaPlan object to display in detail
            client_brief: Client brief for context
        """
        st.markdown("---")
        st.subheader(f"📋 Detailed View: {plan.title}")
        
        # Plan overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Budget",
                f"${plan.total_budget:,.2f}",
                delta=f"${plan.total_budget - client_brief.get('budget', 0):,.2f}" if 'budget' in client_brief else None
            )
        
        with col2:
            st.metric(
                "Total Impressions",
                f"{plan.estimated_impressions:,}",
                help="Estimated total impressions across all formats"
            )
        
        with col3:
            st.metric(
                "Estimated Reach",
                f"{plan.estimated_reach:,}",
                help="Estimated unique users reached"
            )
        
        with col4:
            avg_cpm = sum(alloc.cpm for alloc in plan.allocations) / len(plan.allocations) if plan.allocations else 0
            st.metric(
                "Average CPM",
                f"${avg_cpm:.2f}",
                help="Average cost per thousand impressions"
            )
        
        # Detailed allocations table
        st.subheader("💰 Budget Allocation Details")
        
        allocation_data = []
        for allocation in plan.allocations:
            percentage = (allocation.budget_allocation / plan.total_budget * 100) if plan.total_budget > 0 else 0
            
            allocation_data.append({
                'Format': allocation.format_name,
                'Budget': f"${allocation.budget_allocation:,.2f}",
                'Percentage': f"{percentage:.1f}%",
                'CPM': f"${allocation.cpm:.2f}",
                'Est. Impressions': f"{allocation.estimated_impressions:,}",
                'Recommended Sites': ', '.join(allocation.recommended_sites[:3]) + ('...' if len(allocation.recommended_sites) > 3 else ''),
                'Notes': allocation.notes[:100] + ('...' if len(allocation.notes) > 100 else '')
            })
        
        allocation_df = pd.DataFrame(allocation_data)
        st.dataframe(allocation_df, use_container_width=True)
        
        # Strategy rationale
        st.subheader("🎯 Strategy Rationale")
        st.write(plan.rationale)
        
        # Expandable sections for additional details
        with st.expander("🔍 Site Recommendations by Format"):
            for allocation in plan.allocations:
                if allocation.recommended_sites:
                    st.write(f"**{allocation.format_name}:**")
                    sites_text = ', '.join(allocation.recommended_sites)
                    st.write(sites_text)
                    st.write("")
        
        with st.expander("📈 Performance Projections"):
            st.write("**Estimated Performance Metrics:**")
            
            # Calculate additional metrics
            total_budget = plan.total_budget
            total_impressions = plan.estimated_impressions
            
            if total_budget > 0 and total_impressions > 0:
                effective_cpm = (total_budget / total_impressions) * 1000
                st.write(f"• Effective CPM: ${effective_cpm:.2f}")
                
                if plan.estimated_reach > 0:
                    frequency = total_impressions / plan.estimated_reach
                    st.write(f"• Average Frequency: {frequency:.1f}")
                    
                    reach_percentage = (plan.estimated_reach / 1000000) * 100  # Assuming 1M population
                    st.write(f"• Estimated Reach %: {reach_percentage:.2f}%")
            
            # Budget efficiency by format
            st.write("**Budget Efficiency by Format:**")
            for allocation in plan.allocations:
                if allocation.budget_allocation > 0:
                    impressions_per_dollar = allocation.estimated_impressions / allocation.budget_allocation
                    st.write(f"• {allocation.format_name}: {impressions_per_dollar:.0f} impressions per $1")
    
    def _render_comparison_summary(self, plans: List[Any], client_brief: Dict[str, Any]):
        """
        Render comparison summary highlighting key differences between plans.
        
        Args:
            plans: List of MediaPlan objects to compare
            client_brief: Client brief for context
        """
        st.markdown("---")
        st.subheader("🔄 Plan Comparison Summary")
        
        # Comparison table
        comparison_data = []
        
        for i, plan in enumerate(plans):
            # Calculate key metrics
            avg_cpm = sum(alloc.cpm for alloc in plan.allocations) / len(plan.allocations) if plan.allocations else 0
            format_count = len(plan.allocations)
            budget_utilization = (plan.total_budget / client_brief.get('budget', plan.total_budget)) * 100
            
            # Find dominant format (highest budget allocation)
            dominant_format = max(plan.allocations, key=lambda x: x.budget_allocation).format_name if plan.allocations else "None"
            
            comparison_data.append({
                'Plan': plan.title,
                'Total Budget': f"${plan.total_budget:,.2f}",
                'Budget Utilization': f"{budget_utilization:.1f}%",
                'Formats Used': format_count,
                'Dominant Format': dominant_format,
                'Est. Impressions': f"{plan.estimated_impressions:,}",
                'Est. Reach': f"{plan.estimated_reach:,}",
                'Avg CPM': f"${avg_cpm:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Key differences highlighting
        st.subheader("🎯 Key Strategic Differences")
        
        if len(plans) >= 2:
            # Compare budget allocations
            st.write("**Budget Allocation Strategies:**")
            
            for i, plan in enumerate(plans):
                format_allocations = {alloc.format_name: alloc.budget_allocation for alloc in plan.allocations}
                top_formats = sorted(format_allocations.items(), key=lambda x: x[1], reverse=True)[:2]
                
                if top_formats:
                    top_format_text = f"{top_formats[0][0]} (${top_formats[0][1]:,.0f})"
                    if len(top_formats) > 1:
                        top_format_text += f", {top_formats[1][0]} (${top_formats[1][1]:,.0f})"
                    
                    st.write(f"• **{plan.title}**: Focuses on {top_format_text}")
            
            # Compare reach vs frequency strategies
            st.write("**Reach vs Frequency Approach:**")
            
            for plan in plans:
                if plan.estimated_reach > 0 and plan.estimated_impressions > 0:
                    frequency = plan.estimated_impressions / plan.estimated_reach
                    
                    if frequency < 2:
                        strategy = "Broad reach, low frequency"
                    elif frequency > 4:
                        strategy = "Targeted reach, high frequency"
                    else:
                        strategy = "Balanced reach and frequency"
                    
                    st.write(f"• **{plan.title}**: {strategy} (Freq: {frequency:.1f})")
        
        # Recommendation based on objectives
        if client_brief.get('objective'):
            st.subheader("💡 Recommendation Based on Objective")
            
            objective = client_brief['objective'].lower()
            
            if 'awareness' in objective or 'reach' in objective:
                # Find plan with highest reach
                best_plan = max(plans, key=lambda x: x.estimated_reach)
                st.success(f"**For {client_brief['objective']}**: Consider **{best_plan.title}** for maximum reach ({best_plan.estimated_reach:,} estimated reach)")
            
            elif 'engagement' in objective or 'frequency' in objective:
                # Find plan with highest frequency
                best_plan = None
                best_frequency = 0
                
                for plan in plans:
                    if plan.estimated_reach > 0:
                        frequency = plan.estimated_impressions / plan.estimated_reach
                        if frequency > best_frequency:
                            best_frequency = frequency
                            best_plan = plan
                
                if best_plan:
                    st.success(f"**For {client_brief['objective']}**: Consider **{best_plan.title}** for higher engagement (Freq: {best_frequency:.1f})")
            
            elif 'conversion' in objective or 'performance' in objective:
                # Find plan with best cost efficiency
                best_plan = min(plans, key=lambda x: sum(alloc.cpm for alloc in x.allocations) / len(x.allocations) if x.allocations else float('inf'))
                avg_cpm = sum(alloc.cpm for alloc in best_plan.allocations) / len(best_plan.allocations) if best_plan.allocations else 0
                st.success(f"**For {client_brief['objective']}**: Consider **{best_plan.title}** for cost efficiency (Avg CPM: ${avg_cpm:.2f})")
    
    def render_plan_export_options(self, plans: List[Any], selected_plan_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Render export options for media plans.
        
        Args:
            plans: List of MediaPlan objects
            selected_plan_index: Index of selected plan for individual export
            
        Returns:
            Dictionary with export actions and data
        """
        if not plans:
            return {}
        
        st.subheader("📥 Export Options")
        
        export_actions = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export selected plan or first plan
            plan_to_export = plans[selected_plan_index] if selected_plan_index is not None else plans[0]
            
            if st.button("📄 Export Selected Plan (CSV)", use_container_width=True):
                export_actions['export_csv'] = self._prepare_plan_csv_export(plan_to_export)
        
        with col2:
            if st.button("📊 Export All Plans (CSV)", use_container_width=True):
                export_actions['export_all_csv'] = self._prepare_all_plans_csv_export(plans)
        
        with col3:
            if st.button("📋 Export Comparison (PDF)", use_container_width=True):
                export_actions['export_pdf'] = True
        
        return export_actions
    
    def _prepare_plan_csv_export(self, plan: Any) -> str:
        """
        Prepare CSV export data for a single plan.
        
        Args:
            plan: MediaPlan object to export
            
        Returns:
            CSV string data
        """
        try:
            # Plan header information
            csv_data = []
            csv_data.append(f"Media Plan Export: {plan.title}")
            csv_data.append(f"Generated: {plan.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            csv_data.append(f"Total Budget: ${plan.total_budget:,.2f}")
            csv_data.append(f"Estimated Impressions: {plan.estimated_impressions:,}")
            csv_data.append(f"Estimated Reach: {plan.estimated_reach:,}")
            csv_data.append("")
            
            # Strategy rationale
            csv_data.append("Strategy Rationale:")
            csv_data.append(f'"{plan.rationale}"')
            csv_data.append("")
            
            # Allocation details header
            csv_data.append("Format,Budget Allocation,Percentage,CPM,Estimated Impressions,Recommended Sites,Notes")
            
            # Allocation data
            for allocation in plan.allocations:
                percentage = (allocation.budget_allocation / plan.total_budget * 100) if plan.total_budget > 0 else 0
                sites = '; '.join(allocation.recommended_sites)
                notes = allocation.notes.replace('"', '""')  # Escape quotes for CSV
                
                csv_data.append(
                    f'"{allocation.format_name}",'
                    f'${allocation.budget_allocation:,.2f},'
                    f'{percentage:.1f}%,'
                    f'${allocation.cpm:.2f},'
                    f'{allocation.estimated_impressions:,},'
                    f'"{sites}",'
                    f'"{notes}"'
                )
            
            return '\n'.join(csv_data)
            
        except Exception as e:
            logger.error(f"Error preparing CSV export: {str(e)}")
            return f"Error preparing export: {str(e)}"
    
    def _prepare_all_plans_csv_export(self, plans: List[Any]) -> str:
        """
        Prepare CSV export data for all plans comparison.
        
        Args:
            plans: List of MediaPlan objects to export
            
        Returns:
            CSV string data
        """
        try:
            csv_data = []
            csv_data.append("Media Plans Comparison Export")
            csv_data.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            csv_data.append(f"Number of Plans: {len(plans)}")
            csv_data.append("")
            
            # Summary comparison
            csv_data.append("Plan Summary Comparison")
            csv_data.append("Plan Title,Total Budget,Estimated Impressions,Estimated Reach,Formats Used,Average CPM")
            
            for plan in plans:
                avg_cpm = sum(alloc.cpm for alloc in plan.allocations) / len(plan.allocations) if plan.allocations else 0
                format_count = len(plan.allocations)
                
                csv_data.append(
                    f'"{plan.title}",'
                    f'${plan.total_budget:,.2f},'
                    f'{plan.estimated_impressions:,},'
                    f'{plan.estimated_reach:,},'
                    f'{format_count},'
                    f'${avg_cpm:.2f}'
                )
            
            csv_data.append("")
            
            # Detailed allocations for each plan
            for i, plan in enumerate(plans):
                csv_data.append(f"Plan {i+1}: {plan.title}")
                csv_data.append("Format,Budget Allocation,Percentage,CPM,Estimated Impressions")
                
                for allocation in plan.allocations:
                    percentage = (allocation.budget_allocation / plan.total_budget * 100) if plan.total_budget > 0 else 0
                    
                    csv_data.append(
                        f'"{allocation.format_name}",'
                        f'${allocation.budget_allocation:,.2f},'
                        f'{percentage:.1f}%,'
                        f'${allocation.cpm:.2f},'
                        f'{allocation.estimated_impressions:,}'
                    )
                
                csv_data.append("")
            
            return '\n'.join(csv_data)
            
        except Exception as e:
            logger.error(f"Error preparing all plans CSV export: {str(e)}")
            return f"Error preparing export: {str(e)}"
    
    def render_interactive_budget_breakdown(self, plans: List[Any], client_brief: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render interactive budget breakdown with charts and comparison elements.
        
        Args:
            plans: List of MediaPlan objects
            client_brief: Client brief for context
            
        Returns:
            Dictionary with user interactions and selections
        """
        if not plans:
            return {}
        
        st.subheader("📊 Interactive Budget Analysis")
        
        # Plan selection for detailed analysis
        plan_options = [f"{i+1}. {plan.title}" for i, plan in enumerate(plans)]
        selected_plan_name = st.selectbox(
            "Select plan for detailed analysis:",
            options=plan_options,
            key="budget_analysis_plan_selector"
        )
        
        selected_plan_index = int(selected_plan_name.split('.')[0]) - 1
        selected_plan = plans[selected_plan_index]
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["💰 Budget Breakdown", "📈 Performance Metrics", "🔄 Plan Comparison", "🎯 Optimization Insights"])
        
        with tab1:
            self._render_budget_breakdown_charts(selected_plan, client_brief)
        
        with tab2:
            self._render_performance_metrics_charts(selected_plan, plans)
        
        with tab3:
            self._render_comparison_charts(plans, client_brief)
        
        with tab4:
            self._render_optimization_insights(selected_plan, plans, client_brief)
        
        return {
            'selected_plan_index': selected_plan_index,
            'selected_plan': selected_plan
        }
    
    def _render_budget_breakdown_charts(self, plan: Any, client_brief: Dict[str, Any]):
        """
        Render budget breakdown charts for a specific plan.
        
        Args:
            plan: MediaPlan object to analyze
            client_brief: Client brief for context
        """
        st.subheader(f"💰 Budget Breakdown: {plan.title}")
        
        # Prepare data for charts
        allocation_data = []
        for allocation in plan.allocations:
            allocation_data.append({
                'Format': allocation.format_name,
                'Budget': allocation.budget_allocation,
                'Percentage': (allocation.budget_allocation / plan.total_budget * 100) if plan.total_budget > 0 else 0,
                'CPM': allocation.cpm,
                'Impressions': allocation.estimated_impressions
            })
        
        if not allocation_data:
            st.warning("No allocation data available for this plan.")
            return
        
        df = pd.DataFrame(allocation_data)
        
        # Budget allocation pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Budget Distribution by Format**")
            
            # Create pie chart data
            fig_data = {
                'labels': df['Format'].tolist(),
                'values': df['Budget'].tolist(),
                'type': 'pie',
                'textinfo': 'label+percent',
                'textposition': 'auto',
                'hovertemplate': '<b>%{label}</b><br>Budget: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
            }
            
            # Display using Streamlit's built-in chart (fallback to bar chart if pie not available)
            try:
                import plotly.express as px
                fig = px.pie(df, values='Budget', names='Format', 
                           title=f"Budget Allocation - {plan.title}")
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                # Fallback to bar chart if plotly not available
                st.bar_chart(df.set_index('Format')['Budget'])
        
        with col2:
            st.write("**Impressions by Format**")
            
            try:
                import plotly.express as px
                fig = px.bar(df, x='Format', y='Impressions',
                           title="Estimated Impressions by Format",
                           color='Impressions',
                           color_continuous_scale='Blues')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                # Fallback to Streamlit bar chart
                st.bar_chart(df.set_index('Format')['Impressions'])
        
        # Interactive budget allocation table
        st.write("**Interactive Budget Details**")
        
        # Add interactive elements
        show_percentages = st.checkbox("Show percentages", value=True, key="show_percentages_budget")
        show_cpm = st.checkbox("Show CPM details", value=True, key="show_cpm_budget")
        
        # Prepare display dataframe
        display_df = df.copy()
        
        if show_percentages:
            display_df['Budget Display'] = display_df.apply(
                lambda row: f"${row['Budget']:,.2f} ({row['Percentage']:.1f}%)", axis=1
            )
        else:
            display_df['Budget Display'] = display_df['Budget'].apply(lambda x: f"${x:,.2f}")
        
        if show_cpm:
            display_df['CPM Display'] = display_df['CPM'].apply(lambda x: f"${x:.2f}")
        
        # Select columns to display
        display_columns = ['Format', 'Budget Display', 'Impressions']
        if show_cpm:
            display_columns.insert(2, 'CPM Display')
        
        # Rename columns for display
        column_mapping = {
            'Budget Display': 'Budget',
            'CPM Display': 'CPM',
            'Impressions': 'Est. Impressions'
        }
        
        final_df = display_df[display_columns].rename(columns=column_mapping)
        st.dataframe(final_df, use_container_width=True)
        
        # Budget utilization metrics
        st.write("**Budget Utilization Analysis**")
        
        total_budget = client_brief.get('budget', plan.total_budget)
        utilization = (plan.total_budget / total_budget * 100) if total_budget > 0 else 100
        remaining_budget = total_budget - plan.total_budget
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Budget Utilization",
                f"{utilization:.1f}%",
                delta=f"${plan.total_budget - total_budget:,.2f}" if total_budget != plan.total_budget else None
            )
        
        with col2:
            st.metric(
                "Remaining Budget",
                f"${remaining_budget:,.2f}",
                delta=f"{(remaining_budget/total_budget*100):.1f}% unused" if remaining_budget > 0 else "Fully utilized"
            )
        
        with col3:
            avg_cpm = df['CPM'].mean() if not df.empty else 0
            st.metric(
                "Average CPM",
                f"${avg_cpm:.2f}",
                help="Weighted average CPM across all formats"
            )
    
    def _render_performance_metrics_charts(self, plan: Any, all_plans: List[Any]):
        """
        Render performance metrics and projections charts.
        
        Args:
            plan: Selected MediaPlan object
            all_plans: All available plans for comparison
        """
        st.subheader(f"📈 Performance Metrics: {plan.title}")
        
        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        total_impressions = plan.estimated_impressions
        total_reach = plan.estimated_reach
        frequency = total_impressions / total_reach if total_reach > 0 else 0
        effective_cpm = (plan.total_budget / total_impressions * 1000) if total_impressions > 0 else 0
        
        with col1:
            st.metric("Total Impressions", f"{total_impressions:,}")
        
        with col2:
            st.metric("Estimated Reach", f"{total_reach:,}")
        
        with col3:
            st.metric("Average Frequency", f"{frequency:.2f}")
        
        with col4:
            st.metric("Effective CPM", f"${effective_cpm:.2f}")
        
        # Performance comparison with other plans
        if len(all_plans) > 1:
            st.write("**Performance Comparison Across Plans**")
            
            comparison_data = []
            for i, p in enumerate(all_plans):
                p_frequency = p.estimated_impressions / p.estimated_reach if p.estimated_reach > 0 else 0
                p_effective_cpm = (p.total_budget / p.estimated_impressions * 1000) if p.estimated_impressions > 0 else 0
                
                comparison_data.append({
                    'Plan': p.title,
                    'Impressions': p.estimated_impressions,
                    'Reach': p.estimated_reach,
                    'Frequency': p_frequency,
                    'Effective CPM': p_effective_cpm,
                    'Selected': p.plan_id == plan.plan_id
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Highlight selected plan
            def highlight_selected(row):
                return ['background-color: #e6f3ff' if row['Selected'] else '' for _ in row]
            
            display_df = comparison_df.drop('Selected', axis=1).copy()
            display_df['Impressions'] = display_df['Impressions'].apply(lambda x: f"{x:,}")
            display_df['Reach'] = display_df['Reach'].apply(lambda x: f"{x:,}")
            display_df['Frequency'] = display_df['Frequency'].apply(lambda x: f"{x:.2f}")
            display_df['Effective CPM'] = display_df['Effective CPM'].apply(lambda x: f"${x:.2f}")
            
            styled_df = display_df.style.apply(highlight_selected, axis=1, subset=None)
            st.dataframe(styled_df, use_container_width=True)
        
        # Reach vs Frequency analysis
        st.write("**Reach vs Frequency Analysis**")
        
        if frequency > 0:
            # Categorize strategy
            if frequency < 2:
                strategy_type = "Broad Reach Strategy"
                strategy_color = "🟢"
                strategy_desc = "Low frequency, maximum reach - ideal for awareness campaigns"
            elif frequency > 4:
                strategy_type = "High Frequency Strategy"
                strategy_color = "🔴"
                strategy_desc = "High frequency, targeted reach - ideal for conversion campaigns"
            else:
                strategy_type = "Balanced Strategy"
                strategy_color = "🟡"
                strategy_desc = "Balanced reach and frequency - versatile approach"
            
            st.info(f"{strategy_color} **{strategy_type}**: {strategy_desc}")
            
            # Frequency distribution by format
            format_frequencies = []
            for allocation in plan.allocations:
                if allocation.estimated_impressions > 0 and total_reach > 0:
                    format_contribution = allocation.estimated_impressions / total_impressions
                    format_frequency = frequency * format_contribution
                    format_frequencies.append({
                        'Format': allocation.format_name,
                        'Frequency Contribution': format_frequency,
                        'Impression Share': format_contribution * 100
                    })
            
            if format_frequencies:
                freq_df = pd.DataFrame(format_frequencies)
                
                try:
                    import plotly.express as px
                    fig = px.bar(freq_df, x='Format', y='Frequency Contribution',
                               title="Frequency Contribution by Format",
                               color='Impression Share',
                               color_continuous_scale='Viridis')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(freq_df.set_index('Format')['Frequency Contribution'])
        
        # Cost efficiency analysis
        st.write("**Cost Efficiency Analysis**")
        
        efficiency_data = []
        for allocation in plan.allocations:
            if allocation.budget_allocation > 0:
                impressions_per_dollar = allocation.estimated_impressions / allocation.budget_allocation
                cost_per_impression = allocation.budget_allocation / allocation.estimated_impressions if allocation.estimated_impressions > 0 else 0
                
                efficiency_data.append({
                    'Format': allocation.format_name,
                    'Impressions per $1': impressions_per_dollar,
                    'Cost per Impression': cost_per_impression,
                    'CPM': allocation.cpm
                })
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Impressions per Dollar**")
                try:
                    import plotly.express as px
                    fig = px.bar(eff_df, x='Format', y='Impressions per $1',
                               title="Cost Efficiency by Format",
                               color='Impressions per $1',
                               color_continuous_scale='RdYlGn')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(eff_df.set_index('Format')['Impressions per $1'])
            
            with col2:
                st.write("**CPM Comparison**")
                try:
                    import plotly.express as px
                    fig = px.bar(eff_df, x='Format', y='CPM',
                               title="CPM by Format",
                               color='CPM',
                               color_continuous_scale='RdYlBu_r')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(eff_df.set_index('Format')['CPM'])
    
    def _render_comparison_charts(self, plans: List[Any], client_brief: Dict[str, Any]):
        """
        Render comparison charts highlighting differences between plans.
        
        Args:
            plans: List of MediaPlan objects to compare
            client_brief: Client brief for context
        """
        st.subheader("🔄 Plan Comparison Analysis")
        
        if len(plans) < 2:
            st.info("Need at least 2 plans for comparison analysis.")
            return
        
        # Overall comparison metrics
        st.write("**Overall Performance Comparison**")
        
        comparison_metrics = []
        for plan in plans:
            frequency = plan.estimated_impressions / plan.estimated_reach if plan.estimated_reach > 0 else 0
            effective_cpm = (plan.total_budget / plan.estimated_impressions * 1000) if plan.estimated_impressions > 0 else 0
            format_count = len(plan.allocations)
            
            comparison_metrics.append({
                'Plan': plan.title,
                'Budget': plan.total_budget,
                'Impressions': plan.estimated_impressions,
                'Reach': plan.estimated_reach,
                'Frequency': frequency,
                'Effective CPM': effective_cpm,
                'Formats Used': format_count
            })
        
        metrics_df = pd.DataFrame(comparison_metrics)
        
        # Multi-metric comparison chart
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Budget Comparison', 'Reach vs Impressions', 'Frequency Comparison', 'CPM Efficiency'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Budget comparison
            fig.add_trace(
                go.Bar(x=metrics_df['Plan'], y=metrics_df['Budget'], name='Budget', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Reach vs Impressions scatter
            fig.add_trace(
                go.Scatter(x=metrics_df['Reach'], y=metrics_df['Impressions'], 
                          mode='markers+text', text=metrics_df['Plan'],
                          textposition="top center", name='Reach vs Impressions',
                          marker=dict(size=10, color='orange')),
                row=1, col=2
            )
            
            # Frequency comparison
            fig.add_trace(
                go.Bar(x=metrics_df['Plan'], y=metrics_df['Frequency'], name='Frequency', marker_color='lightgreen'),
                row=2, col=1
            )
            
            # CPM efficiency
            fig.add_trace(
                go.Bar(x=metrics_df['Plan'], y=metrics_df['Effective CPM'], name='Effective CPM', marker_color='lightcoral'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Comprehensive Plan Comparison")
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            # Fallback to individual charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Budget Comparison**")
                st.bar_chart(metrics_df.set_index('Plan')['Budget'])
                
                st.write("**Frequency Comparison**")
                st.bar_chart(metrics_df.set_index('Plan')['Frequency'])
            
            with col2:
                st.write("**Reach Comparison**")
                st.bar_chart(metrics_df.set_index('Plan')['Reach'])
                
                st.write("**CPM Comparison**")
                st.bar_chart(metrics_df.set_index('Plan')['Effective CPM'])
        
        # Format allocation comparison
        st.write("**Format Allocation Comparison**")
        
        # Collect all unique formats across plans
        all_formats = set()
        for plan in plans:
            for allocation in plan.allocations:
                all_formats.add(allocation.format_name)
        
        all_formats = sorted(list(all_formats))
        
        # Create allocation comparison matrix
        allocation_matrix = []
        for plan in plans:
            plan_allocations = {alloc.format_name: alloc.budget_allocation for alloc in plan.allocations}
            
            row = {'Plan': plan.title}
            for format_name in all_formats:
                row[format_name] = plan_allocations.get(format_name, 0)
            
            allocation_matrix.append(row)
        
        allocation_df = pd.DataFrame(allocation_matrix)
        
        # Display as heatmap-style table
        try:
            import plotly.express as px
            
            # Prepare data for heatmap
            heatmap_data = allocation_df.set_index('Plan')[all_formats]
            
            fig = px.imshow(heatmap_data.values,
                          labels=dict(x="Format", y="Plan", color="Budget Allocation"),
                          x=all_formats,
                          y=heatmap_data.index,
                          color_continuous_scale='Blues',
                          title="Budget Allocation Heatmap")
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            # Fallback to table display
            st.dataframe(allocation_df, use_container_width=True)
        
        # Highlight key differences
        st.write("**Key Strategic Differences**")
        
        differences = []
        
        # Budget range analysis
        budgets = [plan.total_budget for plan in plans]
        budget_range = max(budgets) - min(budgets)
        if budget_range > 1000:
            differences.append(f"Budget variation: ${budget_range:,.0f} difference between highest and lowest")
        
        # Format diversity analysis
        format_counts = [len(plan.allocations) for plan in plans]
        if max(format_counts) - min(format_counts) > 0:
            differences.append(f"Format diversity: {min(format_counts)}-{max(format_counts)} formats used across plans")
        
        # Reach strategy analysis
        frequencies = [plan.estimated_impressions / plan.estimated_reach if plan.estimated_reach > 0 else 0 for plan in plans]
        if max(frequencies) - min(frequencies) > 1:
            differences.append(f"Frequency strategy: {min(frequencies):.1f}-{max(frequencies):.1f} frequency range")
        
        for diff in differences:
            st.info(f"• {diff}")
    
    def _render_optimization_insights(self, selected_plan: Any, all_plans: List[Any], client_brief: Dict[str, Any]):
        """
        Render optimization insights and recommendations.
        
        Args:
            selected_plan: Currently selected MediaPlan
            all_plans: All available plans
            client_brief: Client brief for context
        """
        st.subheader("🎯 Optimization Insights")
        
        # Performance scoring
        st.write("**Plan Performance Scoring**")
        
        scores = []
        for plan in all_plans:
            score_data = self._calculate_plan_score(plan, client_brief)
            score_data['Plan'] = plan.title
            score_data['Selected'] = plan.plan_id == selected_plan.plan_id
            scores.append(score_data)
        
        scores_df = pd.DataFrame(scores)
        
        # Display scores with highlighting
        def highlight_selected_score(row):
            return ['background-color: #e6f3ff' if row['Selected'] else '' for _ in row]
        
        display_scores = scores_df.drop('Selected', axis=1)
        styled_scores = display_scores.style.apply(highlight_selected_score, axis=1, subset=None)
        st.dataframe(styled_scores, use_container_width=True)
        
        # Recommendations based on objective
        st.write("**Optimization Recommendations**")
        
        objective = client_brief.get('objective', '').lower()
        budget = client_brief.get('budget', selected_plan.total_budget)
        
        recommendations = self._generate_optimization_recommendations(selected_plan, all_plans, objective, budget)
        
        for rec_type, recommendation in recommendations.items():
            if recommendation:
                if rec_type == 'strength':
                    st.success(f"✅ **Strength**: {recommendation}")
                elif rec_type == 'improvement':
                    st.warning(f"⚠️ **Improvement Opportunity**: {recommendation}")
                elif rec_type == 'alternative':
                    st.info(f"💡 **Alternative Consideration**: {recommendation}")
        
        # Budget optimization suggestions
        st.write("**Budget Optimization Analysis**")
        
        total_budget = client_brief.get('budget', selected_plan.total_budget)
        utilization = (selected_plan.total_budget / total_budget * 100) if total_budget > 0 else 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Budget Utilization", f"{utilization:.1f}%")
            
            if utilization < 90:
                remaining = total_budget - selected_plan.total_budget
                st.info(f"💰 **Opportunity**: ${remaining:,.0f} remaining budget could increase reach by ~{(remaining/selected_plan.total_budget*100):.0f}%")
            elif utilization > 100:
                overage = selected_plan.total_budget - total_budget
                st.warning(f"⚠️ **Budget Exceeded**: Plan exceeds budget by ${overage:,.0f}")
        
        with col2:
            # Calculate efficiency metrics
            if selected_plan.estimated_impressions > 0:
                cost_per_impression = selected_plan.total_budget / selected_plan.estimated_impressions
                st.metric("Cost per Impression", f"${cost_per_impression:.4f}")
                
                # Compare with other plans
                other_costs = []
                for plan in all_plans:
                    if plan.plan_id != selected_plan.plan_id and plan.estimated_impressions > 0:
                        other_cost = plan.total_budget / plan.estimated_impressions
                        other_costs.append(other_cost)
                
                if other_costs:
                    avg_other_cost = sum(other_costs) / len(other_costs)
                    efficiency_vs_others = ((avg_other_cost - cost_per_impression) / avg_other_cost * 100)
                    
                    if efficiency_vs_others > 5:
                        st.success(f"📈 {efficiency_vs_others:.1f}% more efficient than other plans")
                    elif efficiency_vs_others < -5:
                        st.warning(f"📉 {abs(efficiency_vs_others):.1f}% less efficient than other plans")
        
        # Format optimization insights
        st.write("**Format Mix Analysis**")
        
        format_insights = self._analyze_format_mix(selected_plan, all_plans)
        
        for insight in format_insights:
            st.info(f"• {insight}")
    
    def _calculate_plan_score(self, plan: Any, client_brief: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance scores for a media plan.
        
        Args:
            plan: MediaPlan object to score
            client_brief: Client brief for context
            
        Returns:
            Dictionary with various performance scores
        """
        scores = {}
        
        # Budget efficiency score (0-100)
        if plan.estimated_impressions > 0:
            cost_per_impression = plan.total_budget / plan.estimated_impressions
            # Lower cost per impression = higher score
            scores['Cost Efficiency'] = max(0, min(100, (0.001 - cost_per_impression) * 100000))
        else:
            scores['Cost Efficiency'] = 0
        
        # Reach score (0-100)
        # Normalize based on budget - higher reach per dollar = higher score
        if plan.total_budget > 0:
            reach_per_dollar = plan.estimated_reach / plan.total_budget
            scores['Reach Efficiency'] = min(100, reach_per_dollar * 1000)
        else:
            scores['Reach Efficiency'] = 0
        
        # Format diversity score (0-100)
        format_count = len(plan.allocations)
        scores['Format Diversity'] = min(100, format_count * 25)  # Max score at 4+ formats
        
        # Budget utilization score (0-100)
        target_budget = client_brief.get('budget', plan.total_budget)
        if target_budget > 0:
            utilization = plan.total_budget / target_budget
            # Optimal utilization is 90-100%
            if 0.9 <= utilization <= 1.0:
                scores['Budget Utilization'] = 100
            elif utilization > 1.0:
                scores['Budget Utilization'] = max(0, 100 - (utilization - 1.0) * 200)
            else:
                scores['Budget Utilization'] = utilization * 100
        else:
            scores['Budget Utilization'] = 100
        
        # Overall score
        scores['Overall Score'] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _generate_optimization_recommendations(self, selected_plan: Any, all_plans: List[Any], 
                                            objective: str, budget: float) -> Dict[str, str]:
        """
        Generate optimization recommendations based on plan analysis.
        
        Args:
            selected_plan: Currently selected plan
            all_plans: All available plans
            objective: Campaign objective
            budget: Available budget
            
        Returns:
            Dictionary with recommendation types and messages
        """
        recommendations = {}
        
        # Analyze selected plan characteristics
        frequency = selected_plan.estimated_impressions / selected_plan.estimated_reach if selected_plan.estimated_reach > 0 else 0
        utilization = (selected_plan.total_budget / budget * 100) if budget > 0 else 100
        format_count = len(selected_plan.allocations)
        
        # Objective-based recommendations
        if 'awareness' in objective or 'reach' in objective:
            if frequency > 3:
                recommendations['improvement'] = "Consider reducing frequency to maximize reach for awareness campaigns"
            elif selected_plan.estimated_reach == max(plan.estimated_reach for plan in all_plans):
                recommendations['strength'] = "This plan maximizes reach, ideal for awareness objectives"
        
        elif 'engagement' in objective or 'frequency' in objective:
            if frequency < 2:
                recommendations['improvement'] = "Consider increasing frequency for better engagement"
            elif frequency > 5:
                recommendations['strength'] = "High frequency approach supports engagement objectives"
        
        elif 'conversion' in objective or 'performance' in objective:
            avg_cpm = sum(alloc.cpm for alloc in selected_plan.allocations) / len(selected_plan.allocations) if selected_plan.allocations else 0
            other_avg_cpms = []
            for plan in all_plans:
                if plan.plan_id != selected_plan.plan_id and plan.allocations:
                    other_avg_cpm = sum(alloc.cpm for alloc in plan.allocations) / len(plan.allocations)
                    other_avg_cpms.append(other_avg_cpm)
            
            if other_avg_cpms and avg_cpm <= min(other_avg_cpms):
                recommendations['strength'] = "Most cost-efficient plan for performance objectives"
        
        # Budget utilization recommendations
        if utilization < 85:
            remaining = budget - selected_plan.total_budget
            recommendations['improvement'] = f"${remaining:,.0f} remaining budget could boost performance"
        elif utilization > 105:
            overage = selected_plan.total_budget - budget
            recommendations['improvement'] = f"Plan exceeds budget by ${overage:,.0f} - consider optimization"
        
        # Format diversity recommendations
        if format_count < 2:
            recommendations['improvement'] = "Consider diversifying across more ad formats for better reach"
        elif format_count > 4:
            recommendations['improvement'] = "Many formats used - ensure each adds meaningful value"
        
        # Alternative plan recommendations
        if len(all_plans) > 1:
            # Find most different plan
            best_alternative = None
            max_difference = 0
            
            for plan in all_plans:
                if plan.plan_id != selected_plan.plan_id:
                    # Calculate difference score
                    reach_diff = abs(plan.estimated_reach - selected_plan.estimated_reach)
                    budget_diff = abs(plan.total_budget - selected_plan.total_budget)
                    format_diff = abs(len(plan.allocations) - len(selected_plan.allocations))
                    
                    total_diff = reach_diff + budget_diff + format_diff
                    
                    if total_diff > max_difference:
                        max_difference = total_diff
                        best_alternative = plan
            
            if best_alternative:
                recommendations['alternative'] = f"Consider '{best_alternative.title}' for a different strategic approach"
        
        return recommendations
    
    def _analyze_format_mix(self, plan: Any, all_plans: List[Any]) -> List[str]:
        """
        Analyze format mix and provide insights.
        
        Args:
            plan: MediaPlan to analyze
            all_plans: All plans for comparison
            
        Returns:
            List of format mix insights
        """
        insights = []
        
        if not plan.allocations:
            return ["No format allocations found"]
        
        # Analyze allocation distribution
        allocations = [(alloc.format_name, alloc.budget_allocation) for alloc in plan.allocations]
        allocations.sort(key=lambda x: x[1], reverse=True)
        
        total_budget = plan.total_budget
        
        # Check for dominant format
        if allocations[0][1] / total_budget > 0.6:
            insights.append(f"Heavy focus on {allocations[0][0]} ({allocations[0][1]/total_budget*100:.1f}% of budget)")
        
        # Check for balanced distribution
        if len(allocations) >= 3:
            top_3_share = sum(alloc[1] for alloc in allocations[:3]) / total_budget
            if top_3_share < 0.8:
                insights.append("Well-diversified format mix reduces risk")
        
        # Compare with other plans
        format_names = {alloc.format_name for alloc in plan.allocations}
        
        for other_plan in all_plans:
            if other_plan.plan_id != plan.plan_id:
                other_formats = {alloc.format_name for alloc in other_plan.allocations}
                
                unique_to_this = format_names - other_formats
                if unique_to_this:
                    insights.append(f"Unique formats: {', '.join(unique_to_this)}")
                    break
        
        # CPM analysis
        cpms = [alloc.cpm for alloc in plan.allocations]
        if cpms:
            cpm_range = max(cpms) - min(cpms)
            if cpm_range > 10:
                insights.append(f"Wide CPM range (${min(cpms):.2f}-${max(cpms):.2f}) - mix of premium and cost-effective formats")
        
        return insights


class PlanExportComponent:
    """
    Component for exporting and saving media plans in various formats.
    
    Provides CSV export, PDF report generation, and plan persistence functionality
    with comprehensive error handling and validation.
    """
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """
        Initialize the PlanExportComponent.
        
        Args:
            data_manager: DataManager instance for accessing market data
        """
        self.data_manager = data_manager
    
    def render_export_interface(self, plans: List[Any], client_brief: Dict[str, Any], 
                              selected_plan_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Render the complete export interface with all export options.
        
        Args:
            plans: List of MediaPlan objects
            client_brief: Client brief information
            selected_plan_index: Index of selected plan for individual export
            
        Returns:
            Dictionary with export actions and generated content
        """
        if not plans:
            st.warning("No plans available for export.")
            return {}
        
        st.subheader("📥 Export & Save Options")
        
        export_results = {}
        
        # Export format selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Export Format**")
            export_format = st.radio(
                "Choose export format:",
                options=["CSV (Spreadsheet)", "PDF (Report)", "JSON (Data)"],
                key="export_format_selection"
            )
        
        with col2:
            st.write("**Export Scope**")
            if selected_plan_index is not None and selected_plan_index < len(plans):
                default_scope = f"Selected Plan ({plans[selected_plan_index].title})"
            else:
                default_scope = "All Plans"
            
            export_scope = st.radio(
                "What to export:",
                options=["Selected Plan", "All Plans", "Comparison Summary"],
                index=0 if "Selected" in default_scope else 1,
                key="export_scope_selection"
            )
        
        # Export options
        st.write("**Export Options**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_rationale = st.checkbox("Include strategy rationale", value=True, key="include_rationale")
            include_sites = st.checkbox("Include site recommendations", value=True, key="include_sites")
        
        with col2:
            include_metrics = st.checkbox("Include performance metrics", value=True, key="include_metrics")
            include_charts = st.checkbox("Include charts (PDF only)", value=True, key="include_charts")
        
        with col3:
            include_timestamp = st.checkbox("Include generation timestamp", value=True, key="include_timestamp")
            include_brief = st.checkbox("Include client brief", value=True, key="include_brief")
        
        # Export buttons
        st.write("**Generate Export**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📄 Generate Export", type="primary", use_container_width=True):
                export_results = self._generate_export(
                    plans, client_brief, export_format, export_scope, 
                    selected_plan_index, {
                        'include_rationale': include_rationale,
                        'include_sites': include_sites,
                        'include_metrics': include_metrics,
                        'include_charts': include_charts,
                        'include_timestamp': include_timestamp,
                        'include_brief': include_brief
                    }
                )
        
        with col2:
            if st.button("💾 Save Plans", use_container_width=True):
                export_results['save_action'] = self._save_plans_to_session(plans, client_brief)
        
        with col3:
            if st.button("📧 Email Export", use_container_width=True):
                export_results['email_action'] = self._prepare_email_export(plans, client_brief)
        
        with col4:
            if st.button("🔗 Share Link", use_container_width=True):
                export_results['share_action'] = self._generate_share_link(plans, client_brief)
        
        return export_results
    
    def _generate_export(self, plans: List[Any], client_brief: Dict[str, Any], 
                        export_format: str, export_scope: str, selected_plan_index: Optional[int],
                        options: Dict[str, bool]) -> Dict[str, Any]:
        """
        Generate export content based on selected format and options.
        
        Args:
            plans: List of MediaPlan objects
            client_brief: Client brief information
            export_format: Selected export format
            export_scope: Selected export scope
            selected_plan_index: Index of selected plan
            options: Export options dictionary
            
        Returns:
            Dictionary with export results and download data
        """
        try:
            # Determine which plans to export
            if export_scope == "Selected Plan" and selected_plan_index is not None:
                plans_to_export = [plans[selected_plan_index]]
            elif export_scope == "All Plans":
                plans_to_export = plans
            else:  # Comparison Summary
                plans_to_export = plans
            
            # Generate content based on format
            if "CSV" in export_format:
                return self._generate_csv_export(plans_to_export, client_brief, export_scope, options)
            elif "PDF" in export_format:
                return self._generate_pdf_export(plans_to_export, client_brief, export_scope, options)
            elif "JSON" in export_format:
                return self._generate_json_export(plans_to_export, client_brief, export_scope, options)
            else:
                return {'error': 'Unsupported export format'}
        
        except Exception as e:
            logger.error(f"Error generating export: {str(e)}")
            return {'error': f'Export generation failed: {str(e)}'}
    
    def _generate_csv_export(self, plans: List[Any], client_brief: Dict[str, Any], 
                           export_scope: str, options: Dict[str, bool]) -> Dict[str, Any]:
        """
        Generate CSV export content.
        
        Args:
            plans: Plans to export
            client_brief: Client brief information
            export_scope: Export scope
            options: Export options
            
        Returns:
            Dictionary with CSV content and download info
        """
        try:
            csv_content = []
            
            # Header information
            if options.get('include_timestamp', True):
                csv_content.append(f"Media Plan Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                csv_content.append("")
            
            # Client brief information
            if options.get('include_brief', True):
                csv_content.append("CLIENT BRIEF")
                csv_content.append(f"Brand: {client_brief.get('brand_name', 'N/A')}")
                csv_content.append(f"Budget: ${client_brief.get('budget', 0):,.2f}")
                csv_content.append(f"Country: {client_brief.get('country', 'N/A')}")
                csv_content.append(f"Objective: {client_brief.get('objective', 'N/A')}")
                csv_content.append(f"Campaign Period: {client_brief.get('start_date', 'N/A')} to {client_brief.get('end_date', 'N/A')}")
                csv_content.append("")
            
            # Plans data
            if export_scope == "Comparison Summary":
                csv_content.extend(self._create_comparison_csv(plans, options))
            else:
                for i, plan in enumerate(plans):
                    if len(plans) > 1:
                        csv_content.append(f"PLAN {i+1}: {plan.title}")
                        csv_content.append("")
                    
                    csv_content.extend(self._create_single_plan_csv(plan, options))
                    csv_content.append("")
            
            # Create download
            csv_string = '\n'.join(csv_content)
            
            # Generate filename
            if len(plans) == 1:
                filename = f"media_plan_{plans[0].title.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.csv"
            else:
                filename = f"media_plans_comparison_{datetime.now().strftime('%Y%m%d')}.csv"
            
            # Provide download
            st.download_button(
                label="📥 Download CSV",
                data=csv_string.encode('utf-8'),
                file_name=filename,
                mime='text/csv',
                key="download_csv_button"
            )
            
            st.success(f"✅ CSV export generated successfully! ({len(csv_content)} lines)")
            
            return {
                'success': True,
                'format': 'CSV',
                'filename': filename,
                'content': csv_string,
                'line_count': len(csv_content)
            }
        
        except Exception as e:
            logger.error(f"Error generating CSV export: {str(e)}")
            st.error(f"❌ CSV export failed: {str(e)}")
            return {'error': str(e)}
    
    def _create_single_plan_csv(self, plan: Any, options: Dict[str, bool]) -> List[str]:
        """
        Create CSV content for a single plan.
        
        Args:
            plan: MediaPlan object
            options: Export options
            
        Returns:
            List of CSV lines for the plan
        """
        lines = []
        
        # Plan overview
        lines.append(f"Plan Title: {plan.title}")
        lines.append(f"Total Budget: ${plan.total_budget:,.2f}")
        
        if options.get('include_metrics', True):
            lines.append(f"Estimated Impressions: {plan.estimated_impressions:,}")
            lines.append(f"Estimated Reach: {plan.estimated_reach:,}")
            
            if plan.estimated_reach > 0:
                frequency = plan.estimated_impressions / plan.estimated_reach
                lines.append(f"Average Frequency: {frequency:.2f}")
        
        lines.append("")
        
        # Strategy rationale
        if options.get('include_rationale', True) and plan.rationale:
            lines.append("Strategy Rationale:")
            lines.append(f'"{plan.rationale}"')
            lines.append("")
        
        # Allocations table
        lines.append("BUDGET ALLOCATIONS")
        
        # CSV header
        header = ["Format", "Budget Allocation", "Percentage", "CPM", "Estimated Impressions"]
        if options.get('include_sites', True):
            header.append("Recommended Sites")
        if options.get('include_rationale', True):
            header.append("Notes")
        
        lines.append(",".join(header))
        
        # Allocation rows
        for allocation in plan.allocations:
            percentage = (allocation.budget_allocation / plan.total_budget * 100) if plan.total_budget > 0 else 0
            
            row = [
                f'"{allocation.format_name}"',
                f'${allocation.budget_allocation:,.2f}',
                f'{percentage:.1f}%',
                f'${allocation.cpm:.2f}',
                f'{allocation.estimated_impressions:,}'
            ]
            
            if options.get('include_sites', True):
                sites = '; '.join(allocation.recommended_sites) if allocation.recommended_sites else 'N/A'
                row.append(f'"{sites}"')
            
            if options.get('include_rationale', True):
                notes = allocation.notes.replace('"', '""') if allocation.notes else 'N/A'
                row.append(f'"{notes}"')
            
            lines.append(",".join(row))
        
        return lines
    
    def _create_comparison_csv(self, plans: List[Any], options: Dict[str, bool]) -> List[str]:
        """
        Create CSV content for plan comparison.
        
        Args:
            plans: List of MediaPlan objects
            options: Export options
            
        Returns:
            List of CSV lines for comparison
        """
        lines = []
        
        lines.append("PLAN COMPARISON SUMMARY")
        lines.append("")
        
        # Comparison table header
        header = ["Plan", "Total Budget", "Estimated Impressions", "Estimated Reach", "Formats Used"]
        if options.get('include_metrics', True):
            header.extend(["Average CPM", "Average Frequency", "Budget Efficiency"])
        
        lines.append(",".join(header))
        
        # Comparison rows
        for plan in plans:
            avg_cpm = sum(alloc.cpm for alloc in plan.allocations) / len(plan.allocations) if plan.allocations else 0
            frequency = plan.estimated_impressions / plan.estimated_reach if plan.estimated_reach > 0 else 0
            efficiency = plan.estimated_impressions / plan.total_budget if plan.total_budget > 0 else 0
            
            row = [
                f'"{plan.title}"',
                f'${plan.total_budget:,.2f}',
                f'{plan.estimated_impressions:,}',
                f'{plan.estimated_reach:,}',
                f'{len(plan.allocations)}'
            ]
            
            if options.get('include_metrics', True):
                row.extend([
                    f'${avg_cpm:.2f}',
                    f'{frequency:.2f}',
                    f'{efficiency:.0f}'
                ])
            
            lines.append(",".join(row))
        
        lines.append("")
        
        # Detailed allocations for each plan
        if options.get('include_rationale', True):
            lines.append("DETAILED ALLOCATIONS BY PLAN")
            lines.append("")
            
            for i, plan in enumerate(plans):
                lines.append(f"Plan {i+1}: {plan.title}")
                lines.append("Format,Budget,Percentage,CPM,Impressions")
                
                for allocation in plan.allocations:
                    percentage = (allocation.budget_allocation / plan.total_budget * 100) if plan.total_budget > 0 else 0
                    lines.append(
                        f'"{allocation.format_name}",'
                        f'${allocation.budget_allocation:,.2f},'
                        f'{percentage:.1f}%,'
                        f'${allocation.cpm:.2f},'
                        f'{allocation.estimated_impressions:,}'
                    )
                
                lines.append("")
        
        return lines
    
    def _generate_pdf_export(self, plans: List[Any], client_brief: Dict[str, Any], 
                           export_scope: str, options: Dict[str, bool]) -> Dict[str, Any]:
        """
        Generate PDF export content.
        
        Args:
            plans: Plans to export
            client_brief: Client brief information
            export_scope: Export scope
            options: Export options
            
        Returns:
            Dictionary with PDF generation results
        """
        try:
            # For now, create a comprehensive text report that can be converted to PDF
            # In a full implementation, you would use libraries like reportlab or weasyprint
            
            report_content = []
            
            # Title page
            report_content.append("MEDIA PLAN REPORT")
            report_content.append("=" * 50)
            report_content.append("")
            
            if options.get('include_timestamp', True):
                report_content.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
                report_content.append("")
            
            # Executive summary
            report_content.append("EXECUTIVE SUMMARY")
            report_content.append("-" * 20)
            report_content.append("")
            
            if len(plans) == 1:
                plan = plans[0]
                report_content.append(f"This report presents the '{plan.title}' media plan for {client_brief.get('brand_name', 'the client')}.")
                report_content.append(f"Total investment: ${plan.total_budget:,.2f}")
                report_content.append(f"Estimated reach: {plan.estimated_reach:,} users")
                report_content.append(f"Estimated impressions: {plan.estimated_impressions:,}")
            else:
                report_content.append(f"This report presents {len(plans)} strategic media plan options for {client_brief.get('brand_name', 'the client')}.")
                total_budget_range = f"${min(p.total_budget for p in plans):,.0f} - ${max(p.total_budget for p in plans):,.0f}"
                report_content.append(f"Budget range: {total_budget_range}")
            
            report_content.append("")
            
            # Client brief
            if options.get('include_brief', True):
                report_content.append("CLIENT BRIEF")
                report_content.append("-" * 15)
                report_content.append("")
                report_content.append(f"Brand/Advertiser: {client_brief.get('brand_name', 'N/A')}")
                report_content.append(f"Campaign Budget: ${client_brief.get('budget', 0):,.2f}")
                report_content.append(f"Target Market: {client_brief.get('country', 'N/A')}")
                report_content.append(f"Campaign Objective: {client_brief.get('objective', 'N/A')}")
                report_content.append(f"Campaign Period: {client_brief.get('start_date', 'N/A')} to {client_brief.get('end_date', 'N/A')}")
                report_content.append(f"Planning Mode: {client_brief.get('planning_mode', 'N/A')}")
                report_content.append("")
            
            # Plan details
            for i, plan in enumerate(plans):
                report_content.append(f"PLAN {i+1}: {plan.title.upper()}")
                report_content.append("=" * (len(f"PLAN {i+1}: {plan.title.upper()}")))
                report_content.append("")
                
                # Plan overview
                report_content.append("Overview")
                report_content.append("-" * 8)
                report_content.append(f"Total Budget: ${plan.total_budget:,.2f}")
                
                if options.get('include_metrics', True):
                    report_content.append(f"Estimated Impressions: {plan.estimated_impressions:,}")
                    report_content.append(f"Estimated Reach: {plan.estimated_reach:,}")
                    
                    if plan.estimated_reach > 0:
                        frequency = plan.estimated_impressions / plan.estimated_reach
                        report_content.append(f"Average Frequency: {frequency:.2f}")
                    
                    avg_cpm = sum(alloc.cpm for alloc in plan.allocations) / len(plan.allocations) if plan.allocations else 0
                    report_content.append(f"Average CPM: ${avg_cpm:.2f}")
                
                report_content.append("")
                
                # Strategy rationale
                if options.get('include_rationale', True) and plan.rationale:
                    report_content.append("Strategic Rationale")
                    report_content.append("-" * 18)
                    report_content.append(plan.rationale)
                    report_content.append("")
                
                # Budget allocations
                report_content.append("Budget Allocation")
                report_content.append("-" * 17)
                report_content.append("")
                
                for allocation in plan.allocations:
                    percentage = (allocation.budget_allocation / plan.total_budget * 100) if plan.total_budget > 0 else 0
                    
                    report_content.append(f"• {allocation.format_name}")
                    report_content.append(f"  Budget: ${allocation.budget_allocation:,.2f} ({percentage:.1f}%)")
                    report_content.append(f"  CPM: ${allocation.cpm:.2f}")
                    report_content.append(f"  Estimated Impressions: {allocation.estimated_impressions:,}")
                    
                    if options.get('include_sites', True) and allocation.recommended_sites:
                        sites_text = ', '.join(allocation.recommended_sites[:5])
                        if len(allocation.recommended_sites) > 5:
                            sites_text += f" (and {len(allocation.recommended_sites) - 5} more)"
                        report_content.append(f"  Recommended Sites: {sites_text}")
                    
                    if allocation.notes:
                        report_content.append(f"  Notes: {allocation.notes}")
                    
                    report_content.append("")
                
                if i < len(plans) - 1:
                    report_content.append("\n" + "=" * 50 + "\n")
            
            # Comparison section for multiple plans
            if len(plans) > 1 and export_scope != "Selected Plan":
                report_content.append("PLAN COMPARISON")
                report_content.append("=" * 15)
                report_content.append("")
                
                # Comparison table
                report_content.append("Summary Comparison")
                report_content.append("-" * 18)
                report_content.append("")
                
                for plan in plans:
                    avg_cpm = sum(alloc.cpm for alloc in plan.allocations) / len(plan.allocations) if plan.allocations else 0
                    frequency = plan.estimated_impressions / plan.estimated_reach if plan.estimated_reach > 0 else 0
                    
                    report_content.append(f"• {plan.title}")
                    report_content.append(f"  Budget: ${plan.total_budget:,.2f}")
                    report_content.append(f"  Reach: {plan.estimated_reach:,} | Frequency: {frequency:.2f}")
                    report_content.append(f"  Formats: {len(plan.allocations)} | Avg CPM: ${avg_cpm:.2f}")
                    report_content.append("")
            
            # Create text file for download (PDF generation would require additional libraries)
            report_text = '\n'.join(report_content)
            
            filename = f"media_plan_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            
            st.download_button(
                label="📥 Download Report (Text)",
                data=report_text.encode('utf-8'),
                file_name=filename,
                mime='text/plain',
                key="download_pdf_button"
            )
            
            st.info("📄 PDF generation requires additional setup. Text report generated instead.")
            st.success(f"✅ Report generated successfully! ({len(report_content)} sections)")
            
            return {
                'success': True,
                'format': 'Text Report',
                'filename': filename,
                'content': report_text,
                'section_count': len(report_content)
            }
        
        except Exception as e:
            logger.error(f"Error generating PDF export: {str(e)}")
            st.error(f"❌ Report generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_json_export(self, plans: List[Any], client_brief: Dict[str, Any], 
                            export_scope: str, options: Dict[str, bool]) -> Dict[str, Any]:
        """
        Generate JSON export content.
        
        Args:
            plans: Plans to export
            client_brief: Client brief information
            export_scope: Export scope
            options: Export options
            
        Returns:
            Dictionary with JSON export results
        """
        try:
            import json
            
            export_data = {
                'export_info': {
                    'generated_at': datetime.now().isoformat(),
                    'export_scope': export_scope,
                    'format': 'JSON',
                    'options': options
                }
            }
            
            # Include client brief if requested
            if options.get('include_brief', True):
                export_data['client_brief'] = {
                    'brand_name': client_brief.get('brand_name'),
                    'budget': client_brief.get('budget'),
                    'country': client_brief.get('country'),
                    'objective': client_brief.get('objective'),
                    'start_date': str(client_brief.get('start_date', '')),
                    'end_date': str(client_brief.get('end_date', '')),
                    'planning_mode': client_brief.get('planning_mode'),
                    'notes': client_brief.get('notes', '')
                }
            
            # Convert plans to JSON-serializable format
            plans_data = []
            
            for plan in plans:
                plan_data = {
                    'plan_id': plan.plan_id,
                    'title': plan.title,
                    'total_budget': plan.total_budget,
                    'estimated_impressions': plan.estimated_impressions,
                    'estimated_reach': plan.estimated_reach,
                    'created_at': plan.created_at.isoformat(),
                    'allocations': []
                }
                
                if options.get('include_rationale', True):
                    plan_data['rationale'] = plan.rationale
                
                # Add allocations
                for allocation in plan.allocations:
                    alloc_data = {
                        'format_name': allocation.format_name,
                        'budget_allocation': allocation.budget_allocation,
                        'cpm': allocation.cpm,
                        'estimated_impressions': allocation.estimated_impressions
                    }
                    
                    if options.get('include_sites', True):
                        alloc_data['recommended_sites'] = allocation.recommended_sites
                    
                    if options.get('include_rationale', True):
                        alloc_data['notes'] = allocation.notes
                    
                    plan_data['allocations'].append(alloc_data)
                
                # Add calculated metrics if requested
                if options.get('include_metrics', True):
                    plan_data['calculated_metrics'] = {
                        'average_cpm': sum(alloc.cpm for alloc in plan.allocations) / len(plan.allocations) if plan.allocations else 0,
                        'average_frequency': plan.estimated_impressions / plan.estimated_reach if plan.estimated_reach > 0 else 0,
                        'budget_efficiency': plan.estimated_impressions / plan.total_budget if plan.total_budget > 0 else 0,
                        'format_count': len(plan.allocations)
                    }
                
                plans_data.append(plan_data)
            
            export_data['plans'] = plans_data
            
            # Add comparison data for multiple plans
            if len(plans) > 1:
                export_data['comparison'] = {
                    'plan_count': len(plans),
                    'budget_range': {
                        'min': min(p.total_budget for p in plans),
                        'max': max(p.total_budget for p in plans)
                    },
                    'reach_range': {
                        'min': min(p.estimated_reach for p in plans),
                        'max': max(p.estimated_reach for p in plans)
                    },
                    'impressions_range': {
                        'min': min(p.estimated_impressions for p in plans),
                        'max': max(p.estimated_impressions for p in plans)
                    }
                }
            
            # Convert to JSON string
            json_string = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            # Generate filename
            filename = f"media_plans_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            
            # Provide download
            st.download_button(
                label="📥 Download JSON",
                data=json_string.encode('utf-8'),
                file_name=filename,
                mime='application/json',
                key="download_json_button"
            )
            
            st.success(f"✅ JSON export generated successfully! ({len(plans)} plans)")
            
            return {
                'success': True,
                'format': 'JSON',
                'filename': filename,
                'content': json_string,
                'plan_count': len(plans)
            }
        
        except Exception as e:
            logger.error(f"Error generating JSON export: {str(e)}")
            st.error(f"❌ JSON export failed: {str(e)}")
            return {'error': str(e)}
    
    def _save_plans_to_session(self, plans: List[Any], client_brief: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save plans to session state for persistence.
        
        Args:
            plans: Plans to save
            client_brief: Client brief information
            
        Returns:
            Dictionary with save results
        """
        try:
            # Save to session state
            save_data = {
                'plans': plans,
                'client_brief': client_brief,
                'saved_at': datetime.now().isoformat(),
                'save_id': f"save_{int(datetime.now().timestamp())}"
            }
            
            # Store in session state
            if not hasattr(st.session_state, 'saved_plans'):
                st.session_state.saved_plans = []
            
            st.session_state.saved_plans.append(save_data)
            
            # Keep only last 5 saves to prevent memory issues
            if len(st.session_state.saved_plans) > 5:
                st.session_state.saved_plans = st.session_state.saved_plans[-5:]
            
            st.success(f"✅ Plans saved to session! (ID: {save_data['save_id']})")
            
            return {
                'success': True,
                'save_id': save_data['save_id'],
                'saved_at': save_data['saved_at'],
                'plan_count': len(plans)
            }
        
        except Exception as e:
            logger.error(f"Error saving plans: {str(e)}")
            st.error(f"❌ Save failed: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_email_export(self, plans: List[Any], client_brief: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare email export (placeholder for email integration).
        
        Args:
            plans: Plans to email
            client_brief: Client brief information
            
        Returns:
            Dictionary with email preparation results
        """
        try:
            # Create email content
            subject = f"Media Plan for {client_brief.get('brand_name', 'Client')} - {datetime.now().strftime('%Y-%m-%d')}"
            
            body_lines = [
                f"Dear Team,",
                "",
                f"Please find the media plan(s) for {client_brief.get('brand_name', 'the client')} below:",
                "",
                f"Campaign Budget: ${client_brief.get('budget', 0):,.2f}",
                f"Target Market: {client_brief.get('country', 'N/A')}",
                f"Objective: {client_brief.get('objective', 'N/A')}",
                "",
                "Plan Summary:",
            ]
            
            for i, plan in enumerate(plans):
                body_lines.extend([
                    f"{i+1}. {plan.title}",
                    f"   Budget: ${plan.total_budget:,.2f}",
                    f"   Estimated Reach: {plan.estimated_reach:,}",
                    f"   Formats: {len(plan.allocations)}",
                    ""
                ])
            
            body_lines.extend([
                "Please review and let me know if you have any questions.",
                "",
                "Best regards,",
                "Media Planning Team"
            ])
            
            email_body = '\n'.join(body_lines)
            
            # Display email content for copying
            st.info("📧 Email content prepared. Copy the content below:")
            
            with st.expander("Email Content"):
                st.text_area("Subject:", value=subject, height=50, key="email_subject")
                st.text_area("Body:", value=email_body, height=300, key="email_body")
            
            st.info("💡 Copy the subject and body to your email client, or integrate with your email system.")
            
            return {
                'success': True,
                'subject': subject,
                'body': email_body,
                'plan_count': len(plans)
            }
        
        except Exception as e:
            logger.error(f"Error preparing email: {str(e)}")
            st.error(f"❌ Email preparation failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_share_link(self, plans: List[Any], client_brief: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate shareable link (placeholder for sharing functionality).
        
        Args:
            plans: Plans to share
            client_brief: Client brief information
            
        Returns:
            Dictionary with share link results
        """
        try:
            # Generate a simple share ID (in production, this would be stored in a database)
            share_id = f"share_{int(datetime.now().timestamp())}_{len(plans)}plans"
            
            # In a real implementation, you would:
            # 1. Store the plans data in a database with the share_id
            # 2. Generate a proper URL
            # 3. Set expiration dates
            # 4. Handle permissions
            
            share_url = f"https://your-app-domain.com/shared/{share_id}"
            
            st.info("🔗 Share link generated (demo):")
            st.code(share_url)
            
            st.warning("⚠️ This is a demo link. In production, implement proper sharing with:")
            st.write("• Secure data storage")
            st.write("• Access permissions")
            st.write("• Link expiration")
            st.write("• View tracking")
            
            return {
                'success': True,
                'share_id': share_id,
                'share_url': share_url,
                'plan_count': len(plans)
            }
        
        except Exception as e:
            logger.error(f"Error generating share link: {str(e)}")
            st.error(f"❌ Share link generation failed: {str(e)}")
            return {'error': str(e)}
    
    def render_saved_plans_manager(self) -> Dict[str, Any]:
        """
        Render interface for managing saved plans.
        
        Returns:
            Dictionary with management actions
        """
        if 'saved_plans' not in st.session_state or not st.session_state.saved_plans:
            st.info("No saved plans found.")
            return {}
        
        st.subheader("💾 Saved Plans Manager")
        
        actions = {}
        
        # List saved plans
        for i, save_data in enumerate(st.session_state.saved_plans):
            with st.expander(f"Save {i+1}: {save_data['save_id']} ({len(save_data['plans'])} plans)"):
                st.write(f"**Saved:** {save_data['saved_at']}")
                st.write(f"**Brand:** {save_data['client_brief'].get('brand_name', 'N/A')}")
                st.write(f"**Plans:** {', '.join(plan.title for plan in save_data['plans'])}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"Load", key=f"load_{i}"):
                        actions[f'load_{i}'] = save_data
                
                with col2:
                    if st.button(f"Export", key=f"export_saved_{i}"):
                        actions[f'export_{i}'] = save_data
                
                with col3:
                    if st.button(f"Delete", key=f"delete_{i}"):
                        actions[f'delete_{i}'] = i
        
        return actions
    
    def _display_validation_errors(self, errors: Dict[str, str]):
        """
        Display validation errors to the user with categorized messaging.
        
        Args:
            errors: Dictionary of field names to error messages
        """
        # Categorize errors
        critical_errors = []
        warning_errors = []
        
        for field, error in errors.items():
            if any(keyword in error.lower() for keyword in ['required', 'must be', 'cannot be', 'exceeds']):
                critical_errors.append(error)
            else:
                warning_errors.append(error)
        
        # Display critical errors
        if critical_errors:
            st.error("❌ **Critical Issues - Please Fix:**")
            for error in critical_errors:
                st.error(f"• {error}")
        
        # Display warnings
        if warning_errors:
            st.warning("⚠️ **Warnings - Please Review:**")
            for error in warning_errors:
                st.warning(f"• {error}")
        
        # Provide helpful suggestions
        if errors:
            with st.expander("💡 Need Help?"):
                st.write("**Common Solutions:**")
                
                if any('budget' in error.lower() for error in errors.values()):
                    st.write("• **Budget Issues**: Ensure budget is realistic for your target market and campaign goals")
                    st.write("• **Budget Ranges**: Small campaigns: $1K-$10K, Medium: $10K-$100K, Large: $100K+")
                
                if any('date' in error.lower() for error in errors.values()):
                    st.write("• **Date Issues**: Campaign should start within reasonable timeframe and run for adequate duration")
                    st.write("• **Recommended Duration**: 2-4 weeks for awareness, 4-8 weeks for conversions")
                
                if any('country' in error.lower() or 'market' in error.lower() for error in errors.values()):
                    st.write("• **Market Issues**: Ensure data files are uploaded and contain information for your target market")
                    st.write("• **Data Files**: Check that rate cards and site lists are current and complete")
    
    def _display_budget_validation(self, budget: float, country: str):
        """
        Display real-time budget validation and suggestions.
        
        Args:
            budget: Campaign budget
            country: Selected country
        """
        if budget <= 0:
            return
        
        # Budget range suggestions
        if budget < 5000:
            st.warning(
                f"💡 **Budget Suggestion**: With ${budget:,.2f}, consider focusing on 1-2 high-impact "
                "ad formats for maximum effectiveness."
            )
        elif budget < 25000:
            st.info(
                f"💡 **Budget Suggestion**: ${budget:,.2f} allows for a diverse media mix across "
                "3-4 different ad formats."
            )
        elif budget >= 100000:
            st.success(
                f"💡 **Budget Suggestion**: ${budget:,.2f} enables comprehensive campaigns across "
                "all available formats with optimal reach and frequency."
            )
        
        # Market-specific budget insights
        if country:
            try:
                market_data = self.data_manager.get_market_data(country)
                if market_data.get('available'):
                    total_formats = (
                        len(market_data.get('rate_card', {}).get('impact_formats', {})) +
                        len(market_data.get('rate_card', {}).get('reach_formats', {}))
                    )
                    
                    if total_formats > 0:
                        st.info(
                            f"📊 **Market Info**: {country} has {total_formats} available ad formats. "
                            f"Your budget of ${budget:,.2f} can support multiple format combinations."
                        )
                else:
                    st.warning(f"⚠️ Limited data available for {country}. Plans may be restricted.")
            
            except Exception as e:
                logger.error(f"Error getting market data for budget validation: {str(e)}")


class FormatSelectionComponent:
    """
    Component for manual format selection with rate card information and budget allocation preview.
    """
    
    def __init__(self, data_manager: DataManager):
        """
        Initialize the FormatSelectionComponent.
        
        Args:
            data_manager: DataManager instance for accessing rate card data
        """
        self.data_manager = data_manager
    
    def render(self, country: str, budget: float) -> Dict[str, Any]:
        """
        Render the format selection interface.
        
        Args:
            country: Selected country/market
            budget: Available budget
            
        Returns:
            Dictionary containing selected formats and allocation data
        """
        if not country:
            st.warning("Please select a country first to see available ad formats.")
            return {}
        
        st.subheader("🎨 Manual Format Selection")
        
        try:
            # Get market data with error handling
            market_data = self.data_manager.get_market_data(country)
            
            if not market_data.get('available'):
                self._display_format_selection_error(
                    f"Market Data Unavailable for {country}",
                    f"No rate card or site data found for {country}. This market may not be supported or data files may be incomplete.",
                    country
                )
                return {}
            
            # Get available formats with rates
            impact_formats = market_data.get('rate_card', {}).get('impact_formats', {})
            reach_formats = market_data.get('rate_card', {}).get('reach_formats', {})
            
            if not impact_formats and not reach_formats:
                self._display_format_selection_error(
                    f"No Ad Formats Available for {country}",
                    f"Rate card data exists for {country} but contains no pricing information for ad formats.",
                    country
                )
                return {}
            
            # Display format selection
            selected_formats = {}
            total_estimated_cost = 0.0
            
            # Impact formats
            if impact_formats:
                st.write("**APX Impact Formats**")
                for format_name, rate in impact_formats.items():
                    col1, col2, col3 = st.columns([3, 2, 2])
                    
                    with col1:
                        selected = st.checkbox(
                            f"{format_name}",
                            key=f"impact_{format_name}",
                            value=st.session_state.get(f"impact_{format_name}", False)
                        )
                    
                    with col2:
                        st.write(f"${rate:,.2f} CPM")
                    
                    with col3:
                        if selected:
                            allocation = st.number_input(
                                "Budget %",
                                min_value=1,
                                max_value=100,
                                value=st.session_state.get(f"allocation_impact_{format_name}", 20),
                                key=f"allocation_impact_{format_name}",
                                help=f"Percentage of budget for {format_name}"
                            )
                            
                            estimated_cost = (budget * allocation / 100)
                            estimated_impressions = int(estimated_cost / rate * 1000) if rate > 0 else 0
                            
                            selected_formats[f"impact_{format_name}"] = {
                                'type': 'impact',
                                'name': format_name,
                                'rate': rate,
                                'allocation_percent': allocation,
                                'estimated_cost': estimated_cost,
                                'estimated_impressions': estimated_impressions
                            }
                            
                            total_estimated_cost += estimated_cost
                            st.write(f"~{estimated_impressions:,} impressions")
            
            # Reach formats
            if reach_formats:
                st.write("**Reach Media Formats**")
                for format_name, rate in reach_formats.items():
                    col1, col2, col3 = st.columns([3, 2, 2])
                    
                    with col1:
                        selected = st.checkbox(
                            f"{format_name}",
                            key=f"reach_{format_name}",
                            value=st.session_state.get(f"reach_{format_name}", False)
                        )
                    
                    with col2:
                        st.write(f"${rate:,.2f} CPM")
                    
                    with col3:
                        if selected:
                            allocation = st.number_input(
                                "Budget %",
                                min_value=1,
                                max_value=100,
                                value=st.session_state.get(f"allocation_reach_{format_name}", 20),
                                key=f"allocation_reach_{format_name}",
                                help=f"Percentage of budget for {format_name}"
                            )
                            
                            estimated_cost = (budget * allocation / 100)
                            estimated_impressions = int(estimated_cost / rate * 1000) if rate > 0 else 0
                            
                            selected_formats[f"reach_{format_name}"] = {
                                'type': 'reach',
                                'name': format_name,
                                'rate': rate,
                                'allocation_percent': allocation,
                                'estimated_cost': estimated_cost,
                                'estimated_impressions': estimated_impressions
                            }
                            
                            total_estimated_cost += estimated_cost
                            st.write(f"~{estimated_impressions:,} impressions")
            
            # Budget allocation preview
            if selected_formats:
                self._display_budget_preview(selected_formats, budget, total_estimated_cost)
            
            return {
                'selected_formats': selected_formats,
                'total_estimated_cost': total_estimated_cost,
                'budget_utilization': (total_estimated_cost / budget * 100) if budget > 0 else 0
            }
        
        except FileNotFoundError as e:
            logger.error(f"Data files not found for format selection: {str(e)}")
            self._display_format_selection_error(
                "Data Files Missing",
                "Required rate card or site list files are missing. Cannot load format information.",
                country
            )
            return {}
        except Exception as e:
            logger.error(f"Error rendering format selection: {str(e)}")
            self._display_format_selection_error(
                "Format Loading Error",
                f"Unexpected error loading format data for {country}: {str(e)}",
                country
            )
            return {}
    
    def _display_format_selection_error(self, title: str, message: str, country: str):
        """
        Display user-friendly error messages for format selection issues.
        
        Args:
            title: Error title
            message: Detailed error message
            country: Country code for context
        """
        st.error(f"❌ **{title}**")
        st.error(message)
        
        with st.expander("🔧 Troubleshooting Format Selection"):
            st.write(f"**To resolve format selection issues for {country}:**")
            st.write("")
            st.write("1. **Verify Market Support**: Check if your target market is included in the rate card")
            st.write("2. **Update Data Files**: Ensure rate card contains current pricing for all markets")
            st.write("3. **Check File Format**: Rate card should have market codes as column headers")
            st.write("4. **Validate Data**: Ensure rate card contains numeric CPM values")
            st.write("")
            st.write("**Alternative Solutions:**")
            st.write("• Try selecting a different market to test the system")
            st.write("• Use AI Selection mode which may have fallback options")
            st.write("• Contact your data administrator to update market information")
    
    def _display_budget_preview(self, selected_formats: Dict[str, Any], 
                              total_budget: float, total_estimated_cost: float):
        """
        Display budget allocation preview and validation.
        
        Args:
            selected_formats: Dictionary of selected formats with allocation data
            total_budget: Total available budget
            total_estimated_cost: Sum of all format allocations
        """
        st.subheader("💰 Budget Allocation Preview")
        
        # Calculate total allocation percentage
        total_allocation_percent = sum(
            format_data['allocation_percent'] 
            for format_data in selected_formats.values()
        )
        
        # Budget validation
        budget_utilization = (total_estimated_cost / total_budget * 100) if total_budget > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Allocation",
                f"{total_allocation_percent}%",
                delta=f"{total_allocation_percent - 100}%" if total_allocation_percent != 100 else None
            )
        
        with col2:
            st.metric(
                "Estimated Cost",
                f"${total_estimated_cost:,.2f}",
                delta=f"${total_estimated_cost - total_budget:,.2f}" if total_estimated_cost != total_budget else None
            )
        
        with col3:
            st.metric(
                "Budget Utilization",
                f"{budget_utilization:.1f}%"
            )
        
        # Validation messages
        if total_allocation_percent > 100:
            st.error(f"❌ Total allocation ({total_allocation_percent}%) exceeds 100%. Please adjust allocations.")
        elif total_allocation_percent < 100:
            remaining = 100 - total_allocation_percent
            st.warning(f"⚠️ {remaining}% of budget unallocated. Consider adding more formats or increasing allocations.")
        else:
            st.success("✅ Budget allocation is balanced at 100%.")
        
        # Format breakdown
        if len(selected_formats) > 1:
            st.write("**Format Breakdown:**")
            for format_key, format_data in selected_formats.items():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"• {format_data['name']}")
                
                with col2:
                    st.write(f"{format_data['allocation_percent']}%")
                
                with col3:
                    st.write(f"${format_data['estimated_cost']:,.2f}")


def display_validation_summary(form_data: Dict[str, Any], format_data: Dict[str, Any] = None):
    """
    Display a summary of form validation and readiness for plan generation.
    
    Args:
        form_data: Validated form data
        format_data: Optional format selection data for manual mode
    """
    st.subheader("📋 Campaign Summary")
    
    # Basic campaign info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Brand:** {form_data.get('brand_name', 'N/A')}")
        st.write(f"**Budget:** ${form_data.get('budget', 0):,.2f}")
        st.write(f"**Country:** {form_data.get('country', 'N/A')}")
    
    with col2:
        start_date = form_data.get('start_date')
        end_date = form_data.get('end_date')
        if start_date and end_date:
            campaign_days = (end_date - start_date).days
            st.write(f"**Campaign Period:** {campaign_days} days")
            st.write(f"**Start:** {start_date.strftime('%B %d, %Y')}")
            st.write(f"**End:** {end_date.strftime('%B %d, %Y')}")
        
        st.write(f"**Objective:** {form_data.get('objective', 'N/A')}")
    
    # Planning mode info
    planning_mode = form_data.get('planning_mode', 'AI Selection')
    st.write(f"**Planning Mode:** {planning_mode}")
    
    if planning_mode == 'Manual Selection' and format_data:
        selected_count = len(format_data.get('selected_formats', {}))
        if selected_count > 0:
            st.write(f"**Selected Formats:** {selected_count}")
            budget_util = format_data.get('budget_utilization', 0)
            st.write(f"**Budget Utilization:** {budget_util:.1f}%")
    
    # Additional notes
    notes = form_data.get('notes', '').strip()
    if notes:
        st.write(f"**Notes:** {notes}")