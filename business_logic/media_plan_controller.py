"""
Media Plan Controller - Orchestrates the complete media planning workflow.

This module integrates all components of the AI plan generation system
to provide a unified interface for generating optimized media plans.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from models.data_models import ClientBrief, MediaPlan
from data.manager import DataManager
from .ai_plan_generator import AIPlanGenerator
from .budget_optimizer import BudgetOptimizer, OptimizationStrategy
from .plan_validator import PlanValidator, ValidationResult
from .error_handler import error_handler, RetryConfig, ErrorInfo, ErrorSeverity, ErrorCategory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaPlanController:
    """
    Main controller for the media planning workflow.
    
    Orchestrates data loading, AI plan generation, optimization,
    and validation to produce high-quality media plans.
    """
    
    def __init__(self, data_manager: Optional[DataManager] = None, testing_mode: bool = False):
        """
        Initialize the media plan controller.
        
        Args:
            data_manager: Optional DataManager instance
            testing_mode: Skip OpenAI initialization for testing
        """
        self.data_manager = data_manager or DataManager()
        self.ai_generator = AIPlanGenerator(self.data_manager, skip_openai_init=testing_mode)
        self.budget_optimizer = BudgetOptimizer()
        self.plan_validator = PlanValidator()
        
        logger.info("MediaPlanController initialized")
    
    def generate_plans(self, client_brief: ClientBrief) -> Tuple[bool, List[MediaPlan], str, Optional[Dict[str, Any]]]:
        """
        Generate complete media plans for a client brief with comprehensive error handling.
        
        Args:
            client_brief: Client campaign requirements
            
        Returns:
            Tuple of (success, list of MediaPlan objects, status message, user_notification)
        """
        try:
            logger.info(f"Starting plan generation for {client_brief.brand_name}")
            
            # Step 1: Validate data availability with retry
            def validate_data():
                return self.data_manager.validate_data_freshness()
            
            success, validation_status, error_info = error_handler.retry_with_backoff(
                validate_data, 
                RetryConfig(max_attempts=2, base_delay=1.0),
                "data validation"
            )
            
            if not success:
                error_handler.log_error(error_info, "Data validation")
                notification = error_handler.create_user_notification(error_info)
                return False, [], error_info.user_message, notification
            
            if validation_status['overall_status'] not in ['ready', 'needs_refresh']:
                error_info = ErrorInfo(
                    category=ErrorCategory.DATA_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"Data not ready: {validation_status['overall_status']}",
                    user_message="Required data files are not available or outdated. Please upload current rate cards and site lists.",
                    suggested_action="Upload the latest rate card and site list Excel files in the data management section."
                )
                notification = error_handler.create_user_notification(error_info)
                return False, [], error_info.user_message, notification
            
            # Step 2: Get market data with fallback
            try:
                market_data = self.data_manager.get_market_data(client_brief.country)
                if not market_data.get('available'):
                    available_markets = self.data_manager.get_available_markets()
                    error_info = ErrorInfo(
                        category=ErrorCategory.DATA_ERROR,
                        severity=ErrorSeverity.ERROR,
                        message=f"Market {client_brief.country} not available",
                        user_message=f"Market {client_brief.country} is not currently supported.",
                        suggested_action=f"Please select from available markets: {', '.join(available_markets)}"
                    )
                    notification = error_handler.create_user_notification(error_info)
                    return False, [], error_info.user_message, notification
            except Exception as e:
                error_info = error_handler.classify_error(e, "market data retrieval")
                error_handler.log_error(error_info, "Market data")
                notification = error_handler.create_user_notification(error_info)
                return False, [], error_info.user_message, notification
            
            # Step 3: Generate AI plans with retry and fallback
            def generate_ai_plans():
                return self.ai_generator.generate_multiple_plans(client_brief, count=3)
            
            success, ai_plans, error_info = error_handler.retry_with_backoff(
                generate_ai_plans,
                RetryConfig(max_attempts=3, base_delay=2.0, exponential_backoff=True),
                "AI plan generation"
            )
            
            if not success:
                error_handler.log_error(error_info, "AI Plan Generation")
                
                # Try fallback with simpler request
                if error_info.retry_possible:
                    logger.info("Attempting fallback plan generation...")
                    try:
                        fallback_plans = self._generate_fallback_plans(client_brief, market_data)
                        if fallback_plans:
                            notification = {
                                'type': 'warning',
                                'title': 'AI Service Unavailable',
                                'message': 'Generated basic plans using fallback method. AI service will be restored shortly.',
                                'action': 'Plans generated successfully using backup method.'
                            }
                            return True, fallback_plans, "Plans generated using fallback method due to AI service issues.", notification
                    except Exception as fallback_error:
                        logger.error(f"Fallback generation also failed: {str(fallback_error)}")
                
                notification = error_handler.create_user_notification(error_info)
                return False, [], error_info.user_message, notification
            
            if not ai_plans:
                error_info = ErrorInfo(
                    category=ErrorCategory.API_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message="AI generated no valid plans",
                    user_message="The AI service was unable to generate suitable plans for your requirements.",
                    suggested_action="Try adjusting your budget or format selections, or contact support if the issue persists."
                )
                notification = error_handler.create_user_notification(error_info)
                return False, [], error_info.user_message, notification
            
            logger.info(f"AI generated {len(ai_plans)} initial plans")
            
            # Step 4: Validate generated plans with graceful degradation
            validated_plans = ai_plans
            validation_warnings = []
            
            try:
                # Convert plans back to dict format for validation
                plans_data = {
                    'plans': [
                        {
                            'title': plan.title,
                            'rationale': plan.rationale,
                            'total_budget': plan.total_budget,
                            'estimated_reach': plan.estimated_reach,
                            'estimated_impressions': plan.estimated_impressions,
                            'allocations': [
                                {
                                    'format_name': alloc.format_name,
                                    'budget_allocation': alloc.budget_allocation,
                                    'cpm': alloc.cpm,
                                    'estimated_impressions': alloc.estimated_impressions,
                                    'recommended_sites': alloc.recommended_sites,
                                    'notes': alloc.notes
                                }
                                for alloc in plan.allocations
                            ]
                        }
                        for plan in ai_plans
                    ]
                }
                
                validation_result = self.plan_validator.parse_and_validate_plans(
                    plans_data, client_brief, market_data['rate_card']
                )
                
                if validation_result.parsed_plans:
                    validated_plans = validation_result.parsed_plans
                    if validation_result.total_warnings > 0:
                        validation_warnings.append(f"{validation_result.total_warnings} validation warnings")
                
                logger.info(f"Validation completed: {len(validated_plans)} valid plans")
                
            except Exception as e:
                logger.warning(f"Plan validation failed, using AI plans directly: {str(e)}")
                validation_warnings.append("Plan validation skipped due to technical issues")
            
            # Step 5: Enhance plans with site recommendations (non-critical)
            enhanced_plans = validated_plans
            try:
                enhanced_plans = self._enhance_plans_with_sites(validated_plans, market_data)
                logger.info("Plans enhanced with site recommendations")
            except Exception as e:
                logger.warning(f"Site enhancement failed: {str(e)}")
                validation_warnings.append("Site recommendations may be incomplete")
            
            # Step 6: Final quality check
            final_plans = self._perform_final_quality_check(enhanced_plans, client_brief)
            
            # Prepare success message and notification
            success_message = f"Successfully generated {len(final_plans)} optimized media plans"
            notification = None
            
            if validation_warnings:
                warning_text = "; ".join(validation_warnings)
                success_message += f" (Note: {warning_text})"
                notification = {
                    'type': 'warning',
                    'title': 'Plans Generated with Warnings',
                    'message': f"Plans created successfully. {warning_text}",
                    'dismissible': True
                }
            
            logger.info(f"Plan generation complete: {len(final_plans)} plans delivered")
            return True, final_plans, success_message, notification
            
        except Exception as e:
            # Handle unexpected errors
            error_info = error_handler.classify_error(e, "plan generation workflow")
            error_handler.log_error(error_info, "Plan Generation")
            notification = error_handler.create_user_notification(error_info)
            return False, [], error_info.user_message, notification
    
    def _enhance_plans_with_sites(self, plans: List[MediaPlan], market_data: Dict[str, Any]) -> List[MediaPlan]:
        """
        Enhance plans with specific site recommendations.
        
        Args:
            plans: List of MediaPlan objects
            market_data: Market data with site information
            
        Returns:
            List of enhanced MediaPlan objects
        """
        try:
            enhanced_plans = []
            sites_data = market_data.get('sites', {})
            sites_by_format = sites_data.get('sites_by_format', {})
            
            for plan in plans:
                enhanced_allocations = []
                
                for allocation in plan.allocations:
                    # Get sites for this format
                    format_sites = sites_by_format.get(allocation.format_name, [])
                    
                    if format_sites and not allocation.recommended_sites:
                        # Select top sites based on budget allocation
                        num_sites = min(len(format_sites), max(3, int(allocation.budget_allocation / 5000)))
                        recommended_sites = format_sites[:num_sites]
                    else:
                        recommended_sites = allocation.recommended_sites
                    
                    # Create enhanced allocation
                    enhanced_allocation = allocation
                    enhanced_allocation.recommended_sites = recommended_sites
                    enhanced_allocations.append(enhanced_allocation)
                
                # Create enhanced plan
                enhanced_plan = MediaPlan(
                    plan_id=plan.plan_id,
                    title=plan.title,
                    total_budget=plan.total_budget,
                    allocations=enhanced_allocations,
                    estimated_reach=plan.estimated_reach,
                    estimated_impressions=plan.estimated_impressions,
                    rationale=plan.rationale,
                    created_at=plan.created_at
                )
                
                enhanced_plans.append(enhanced_plan)
            
            return enhanced_plans
            
        except Exception as e:
            logger.warning(f"Error enhancing plans with sites: {str(e)}")
            return plans
    
    def _perform_final_quality_check(self, plans: List[MediaPlan], client_brief: ClientBrief) -> List[MediaPlan]:
        """
        Perform final quality check and filtering on plans.
        
        Args:
            plans: List of MediaPlan objects
            client_brief: Original client brief
            
        Returns:
            List of quality-checked MediaPlan objects
        """
        try:
            quality_plans = []
            
            for plan in plans:
                # Check budget utilization
                budget_utilization = plan.total_budget / client_brief.budget
                if budget_utilization < 0.3:  # Less than 30% utilization
                    logger.warning(f"Plan '{plan.title}' has low budget utilization: {budget_utilization:.1%}")
                    continue
                
                # Check allocation diversity
                if len(plan.allocations) == 0:
                    logger.warning(f"Plan '{plan.title}' has no allocations")
                    continue
                
                # Check for reasonable impressions
                if plan.estimated_impressions <= 0:
                    logger.warning(f"Plan '{plan.title}' has no estimated impressions")
                    continue
                
                quality_plans.append(plan)
            
            # Ensure we have at least one plan
            if not quality_plans and plans:
                logger.warning("No plans passed quality check, returning best available plan")
                quality_plans = [plans[0]]
            
            return quality_plans
            
        except Exception as e:
            logger.warning(f"Error in final quality check: {str(e)}")
            return plans
    
    def validate_inputs(self, client_brief: ClientBrief) -> Tuple[bool, str]:
        """
        Validate client brief inputs before plan generation.
        
        Args:
            client_brief: Client campaign requirements
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Sanitize inputs
            sanitized_brief = self._sanitize_client_brief(client_brief)
            
            # Check required fields
            if not sanitized_brief.brand_name or not sanitized_brief.brand_name.strip():
                return False, "Brand name is required"
            
            if sanitized_brief.budget <= 0:
                return False, "Budget must be positive"
            
            if sanitized_brief.budget > 1000000:  # $1M limit
                return False, "Budget exceeds maximum limit of $1,000,000"
            
            if not sanitized_brief.country or not sanitized_brief.country.strip():
                return False, "Country is required"
            
            # Validate campaign period format
            if sanitized_brief.campaign_period and not self._validate_campaign_period(sanitized_brief.campaign_period):
                return False, "Campaign period format is invalid. Use formats like 'Q1 2024', 'Jan-Mar 2024', or '2024-01-01 to 2024-03-31'"
            
            # Check market availability
            available_markets = self.data_manager.get_available_markets()
            if sanitized_brief.country not in available_markets:
                return False, f"Market {sanitized_brief.country} is not available. Available markets: {', '.join(available_markets)}"
            
            # Validate selected formats if provided
            if sanitized_brief.selected_formats:
                market_data = self.data_manager.get_market_data(sanitized_brief.country)
                available_formats = set()
                
                rate_card = market_data.get('rate_card', {})
                available_formats.update(rate_card.get('impact_formats', {}).keys())
                available_formats.update(rate_card.get('reach_formats', {}).keys())
                
                invalid_formats = set(sanitized_brief.selected_formats) - available_formats
                if invalid_formats:
                    return False, f"Invalid formats selected: {', '.join(invalid_formats)}"
            
            # Validate planning mode
            valid_modes = ['AI', 'Manual']
            if sanitized_brief.planning_mode not in valid_modes:
                return False, f"Planning mode must be one of: {', '.join(valid_modes)}"
            
            return True, "Validation passed"
            
        except Exception as e:
            logger.error(f"Error validating inputs: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def _sanitize_client_brief(self, client_brief: ClientBrief) -> ClientBrief:
        """
        Sanitize client brief inputs to prevent issues.
        
        Args:
            client_brief: Original client brief
            
        Returns:
            Sanitized client brief
        """
        import re
        
        # Sanitize brand name - remove special characters, limit length
        brand_name = client_brief.brand_name.strip() if client_brief.brand_name else ""
        brand_name = re.sub(r'[^\w\s\-&.]', '', brand_name)[:100]
        
        # Sanitize country code - uppercase, alphanumeric only
        country = client_brief.country.strip().upper() if client_brief.country else ""
        country = re.sub(r'[^A-Z0-9]', '', country)[:5]
        
        # Sanitize campaign period
        campaign_period = client_brief.campaign_period.strip() if client_brief.campaign_period else ""
        campaign_period = re.sub(r'[^\w\s\-/.]', '', campaign_period)[:50]
        
        # Sanitize objective
        objective = client_brief.objective.strip() if client_brief.objective else ""
        objective = re.sub(r'[^\w\s\-&.]', '', objective)[:100]
        
        return ClientBrief(
            brand_name=brand_name,
            budget=max(0, client_brief.budget),  # Ensure non-negative
            country=country,
            campaign_period=campaign_period,
            objective=objective,
            planning_mode=client_brief.planning_mode,
            selected_formats=client_brief.selected_formats
        )
    
    def _validate_campaign_period(self, period: str) -> bool:
        """
        Validate campaign period format.
        
        Args:
            period: Campaign period string
            
        Returns:
            True if valid format
        """
        import re
        
        # Common valid formats
        patterns = [
            r'^Q[1-4]\s+\d{4}$',  # Q1 2024
            r'^[A-Za-z]{3}-[A-Za-z]{3}\s+\d{4}$',  # Jan-Mar 2024
            r'^\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}$',  # 2024-01-01 to 2024-03-31
            r'^[A-Za-z]+\s+\d{4}$',  # January 2024
            r'^\d{1,2}/\d{4}$',  # 1/2024
        ]
        
        return any(re.match(pattern, period.strip()) for pattern in patterns)
    
    def get_available_formats(self, country: str) -> Dict[str, List[str]]:
        """
        Get available ad formats for a specific country.
        
        Args:
            country: Country code
            
        Returns:
            Dictionary with impact and reach format lists
        """
        try:
            market_data = self.data_manager.get_market_data(country)
            
            if not market_data.get('available'):
                return {'impact_formats': [], 'reach_formats': []}
            
            rate_card = market_data.get('rate_card', {})
            
            return {
                'impact_formats': list(rate_card.get('impact_formats', {}).keys()),
                'reach_formats': list(rate_card.get('reach_formats', {}).keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting available formats: {str(e)}")
            return {'impact_formats': [], 'reach_formats': []}
    
    def compare_plans(self, plans: List[MediaPlan], client_brief: ClientBrief) -> Dict[str, Any]:
        """
        Compare and rank media plans based on multiple criteria.
        
        Args:
            plans: List of MediaPlan objects to compare
            client_brief: Original client brief for context
            
        Returns:
            Dictionary with comparison results and rankings
        """
        try:
            if not plans:
                return {'error': 'No plans to compare'}
            
            comparison_results = []
            
            for i, plan in enumerate(plans):
                # Calculate metrics for comparison
                budget_efficiency = plan.total_budget / client_brief.budget
                reach_per_dollar = plan.estimated_reach / plan.total_budget if plan.total_budget > 0 else 0
                impressions_per_dollar = plan.estimated_impressions / plan.total_budget if plan.total_budget > 0 else 0
                format_diversity = len(plan.allocations)
                
                # Calculate allocation balance (how evenly budget is distributed)
                if plan.allocations:
                    allocations = [alloc.budget_allocation for alloc in plan.allocations]
                    total_allocation = sum(allocations)
                    if total_allocation > 0:
                        allocation_ratios = [alloc / total_allocation for alloc in allocations]
                        # Calculate Gini coefficient for balance (0 = perfectly balanced)
                        allocation_balance = 1 - (2 * sum((i + 1) * ratio for i, ratio in enumerate(sorted(allocation_ratios))) / len(allocation_ratios) - (len(allocation_ratios) + 1) / len(allocation_ratios))
                    else:
                        allocation_balance = 0
                else:
                    allocation_balance = 0
                
                # Calculate composite score based on objective
                if client_brief.objective and 'reach' in client_brief.objective.lower():
                    # Reach-focused scoring
                    composite_score = (
                        reach_per_dollar * 0.4 +
                        budget_efficiency * 0.3 +
                        format_diversity * 0.2 +
                        allocation_balance * 0.1
                    )
                elif client_brief.objective and 'frequency' in client_brief.objective.lower():
                    # Frequency-focused scoring
                    composite_score = (
                        impressions_per_dollar * 0.4 +
                        budget_efficiency * 0.3 +
                        (1 - allocation_balance) * 0.2 +  # Less diversity for frequency
                        format_diversity * 0.1
                    )
                else:
                    # Balanced scoring
                    composite_score = (
                        reach_per_dollar * 0.25 +
                        impressions_per_dollar * 0.25 +
                        budget_efficiency * 0.25 +
                        format_diversity * 0.15 +
                        allocation_balance * 0.1
                    )
                
                comparison_results.append({
                    'plan_index': i,
                    'plan_title': plan.title,
                    'budget_efficiency': budget_efficiency,
                    'reach_per_dollar': reach_per_dollar,
                    'impressions_per_dollar': impressions_per_dollar,
                    'format_diversity': format_diversity,
                    'allocation_balance': allocation_balance,
                    'composite_score': composite_score,
                    'total_budget': plan.total_budget,
                    'estimated_reach': plan.estimated_reach,
                    'estimated_impressions': plan.estimated_impressions
                })
            
            # Sort by composite score (descending)
            comparison_results.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # Add rankings
            for rank, result in enumerate(comparison_results, 1):
                result['rank'] = rank
                result['recommendation'] = self._get_plan_recommendation(result, client_brief)
            
            # Generate comparison summary
            summary = self._generate_comparison_summary(comparison_results, client_brief)
            
            return {
                'comparison_results': comparison_results,
                'summary': summary,
                'best_plan_index': comparison_results[0]['plan_index'] if comparison_results else None,
                'comparison_criteria': self._get_comparison_criteria(client_brief)
            }
            
        except Exception as e:
            logger.error(f"Error comparing plans: {str(e)}")
            return {'error': f"Plan comparison failed: {str(e)}"}
    
    def _get_plan_recommendation(self, result: Dict[str, Any], client_brief: ClientBrief) -> str:
        """
        Generate recommendation text for a plan based on its metrics.
        
        Args:
            result: Plan comparison result
            client_brief: Original client brief
            
        Returns:
            Recommendation text
        """
        recommendations = []
        
        if result['budget_efficiency'] > 0.9:
            recommendations.append("Excellent budget utilization")
        elif result['budget_efficiency'] < 0.5:
            recommendations.append("Consider increasing budget allocation")
        
        if result['format_diversity'] >= 3:
            recommendations.append("Good format diversity for broad reach")
        elif result['format_diversity'] == 1:
            recommendations.append("Single format focus - good for targeted campaigns")
        
        if result['reach_per_dollar'] > 5:
            recommendations.append("High reach efficiency")
        
        if result['allocation_balance'] > 0.7:
            recommendations.append("Well-balanced budget distribution")
        
        if not recommendations:
            recommendations.append("Solid overall performance")
        
        return "; ".join(recommendations)
    
    def _generate_comparison_summary(self, results: List[Dict[str, Any]], client_brief: ClientBrief) -> str:
        """
        Generate a summary of the plan comparison.
        
        Args:
            results: Sorted comparison results
            client_brief: Original client brief
            
        Returns:
            Summary text
        """
        if not results:
            return "No plans available for comparison"
        
        best_plan = results[0]
        total_plans = len(results)
        
        summary_parts = [
            f"Analyzed {total_plans} media plan{'s' if total_plans > 1 else ''}.",
            f"Best performing plan: '{best_plan['plan_title']}' with a composite score of {best_plan['composite_score']:.2f}."
        ]
        
        # Add objective-specific insights
        if client_brief.objective:
            if 'reach' in client_brief.objective.lower():
                best_reach_plan = max(results, key=lambda x: x['reach_per_dollar'])
                if best_reach_plan != best_plan:
                    summary_parts.append(f"For maximum reach efficiency, consider '{best_reach_plan['plan_title']}'.")
            elif 'frequency' in client_brief.objective.lower():
                best_frequency_plan = max(results, key=lambda x: x['impressions_per_dollar'])
                if best_frequency_plan != best_plan:
                    summary_parts.append(f"For maximum frequency, consider '{best_frequency_plan['plan_title']}'.")
        
        # Add budget insights
        avg_budget_efficiency = sum(r['budget_efficiency'] for r in results) / len(results)
        if avg_budget_efficiency < 0.7:
            summary_parts.append("Consider increasing budget allocations across all plans.")
        
        return " ".join(summary_parts)
    
    def _get_comparison_criteria(self, client_brief: ClientBrief) -> Dict[str, str]:
        """
        Get the criteria used for plan comparison.
        
        Args:
            client_brief: Original client brief
            
        Returns:
            Dictionary of criteria and their descriptions
        """
        criteria = {
            'budget_efficiency': 'Percentage of total budget utilized',
            'reach_per_dollar': 'Estimated reach divided by budget',
            'impressions_per_dollar': 'Estimated impressions divided by budget',
            'format_diversity': 'Number of different ad formats used',
            'allocation_balance': 'How evenly budget is distributed across formats'
        }
        
        if client_brief.objective:
            if 'reach' in client_brief.objective.lower():
                criteria['primary_focus'] = 'Reach efficiency (40% weight in scoring)'
            elif 'frequency' in client_brief.objective.lower():
                criteria['primary_focus'] = 'Impression efficiency (40% weight in scoring)'
            else:
                criteria['primary_focus'] = 'Balanced approach (equal weights)'
        
        return criteria
    
    def _generate_fallback_plans(self, client_brief: ClientBrief, market_data: Dict[str, Any]) -> List[MediaPlan]:
        """
        Generate basic fallback plans when AI service is unavailable.
        
        Args:
            client_brief: Client campaign requirements
            market_data: Market data with available formats
            
        Returns:
            List of basic MediaPlan objects
        """
        try:
            logger.info("Generating fallback plans using rule-based approach")
            
            # Get available formats and rates
            rate_card = market_data.get('rate_card', {})
            impact_formats = rate_card.get('impact_formats', {})
            reach_formats = rate_card.get('reach_formats', {})
            
            all_formats = {}
            all_formats.update(impact_formats)
            all_formats.update(reach_formats)
            
            if not all_formats:
                logger.error("No formats available for fallback generation")
                return []
            
            # Filter by selected formats if specified
            if client_brief.selected_formats:
                available_formats = {
                    name: data for name, data in all_formats.items() 
                    if name in client_brief.selected_formats
                }
            else:
                available_formats = all_formats
            
            if not available_formats:
                logger.error("No valid formats available after filtering")
                return []
            
            # Sort formats by CPM for different strategies
            sorted_formats = sorted(
                available_formats.items(), 
                key=lambda x: x[1].get('cpm', 999)
            )
            
            fallback_plans = []
            
            # Plan 1: Cost-Efficient Strategy (lowest CPM formats)
            plan1_allocations = []
            remaining_budget = client_brief.budget * 0.9  # Use 90% of budget
            
            for format_name, format_data in sorted_formats[:3]:  # Top 3 cheapest
                if remaining_budget <= 0:
                    break
                
                allocation_budget = remaining_budget / min(3, len(sorted_formats))
                cpm = format_data.get('cpm', 50)
                estimated_impressions = int(allocation_budget / cpm * 1000) if cpm > 0 else 0
                
                plan1_allocations.append(FormatAllocation(
                    format_name=format_name,
                    budget_allocation=allocation_budget,
                    cpm=cpm,
                    estimated_impressions=estimated_impressions,
                    recommended_sites=[],
                    notes="Cost-efficient placement"
                ))
                
                remaining_budget -= allocation_budget
            
            if plan1_allocations:
                plan1 = MediaPlan(
                    plan_id="fallback_1",
                    title="Cost-Efficient Strategy",
                    total_budget=sum(alloc.budget_allocation for alloc in plan1_allocations),
                    allocations=plan1_allocations,
                    estimated_reach=int(sum(alloc.estimated_impressions for alloc in plan1_allocations) * 0.3),
                    estimated_impressions=sum(alloc.estimated_impressions for alloc in plan1_allocations),
                    rationale="Maximizes impressions through cost-efficient format selection",
                    created_at=datetime.now()
                )
                fallback_plans.append(plan1)
            
            # Plan 2: Balanced Strategy (mix of formats)
            if len(sorted_formats) >= 2:
                plan2_allocations = []
                budget_per_format = (client_brief.budget * 0.85) / min(len(sorted_formats), 4)
                
                for format_name, format_data in sorted_formats[:4]:  # Up to 4 formats
                    cpm = format_data.get('cpm', 50)
                    estimated_impressions = int(budget_per_format / cpm * 1000) if cpm > 0 else 0
                    
                    plan2_allocations.append(FormatAllocation(
                        format_name=format_name,
                        budget_allocation=budget_per_format,
                        cpm=cpm,
                        estimated_impressions=estimated_impressions,
                        recommended_sites=[],
                        notes="Balanced format mix"
                    ))
                
                plan2 = MediaPlan(
                    plan_id="fallback_2",
                    title="Balanced Multi-Format Strategy",
                    total_budget=sum(alloc.budget_allocation for alloc in plan2_allocations),
                    allocations=plan2_allocations,
                    estimated_reach=int(sum(alloc.estimated_impressions for alloc in plan2_allocations) * 0.25),
                    estimated_impressions=sum(alloc.estimated_impressions for alloc in plan2_allocations),
                    rationale="Diversified approach across multiple ad formats for broad reach",
                    created_at=datetime.now()
                )
                fallback_plans.append(plan2)
            
            # Plan 3: Premium Strategy (higher CPM formats for quality)
            if len(sorted_formats) >= 2:
                plan3_allocations = []
                premium_formats = sorted_formats[-2:]  # Highest CPM formats
                budget_per_format = (client_brief.budget * 0.8) / len(premium_formats)
                
                for format_name, format_data in premium_formats:
                    cpm = format_data.get('cpm', 50)
                    estimated_impressions = int(budget_per_format / cpm * 1000) if cpm > 0 else 0
                    
                    plan3_allocations.append(FormatAllocation(
                        format_name=format_name,
                        budget_allocation=budget_per_format,
                        cpm=cpm,
                        estimated_impressions=estimated_impressions,
                        recommended_sites=[],
                        notes="Premium placement for quality reach"
                    ))
                
                plan3 = MediaPlan(
                    plan_id="fallback_3",
                    title="Premium Quality Strategy",
                    total_budget=sum(alloc.budget_allocation for alloc in plan3_allocations),
                    allocations=plan3_allocations,
                    estimated_reach=int(sum(alloc.estimated_impressions for alloc in plan3_allocations) * 0.4),
                    estimated_impressions=sum(alloc.estimated_impressions for alloc in plan3_allocations),
                    rationale="Focus on premium formats for high-quality audience engagement",
                    created_at=datetime.now()
                )
                fallback_plans.append(plan3)
            
            logger.info(f"Generated {len(fallback_plans)} fallback plans")
            return fallback_plans
            
        except Exception as e:
            logger.error(f"Fallback plan generation failed: {str(e)}")
            return []
    
    def get_offline_capabilities(self) -> Dict[str, Any]:
        """
        Get information about offline/fallback capabilities.
        
        Returns:
            Dictionary with offline capability information
        """
        try:
            # Check data availability
            data_status = self.data_manager.validate_data_freshness()
            data_available = data_status['overall_status'] in ['ready', 'needs_refresh']
            
            # Check available markets
            available_markets = []
            if data_available:
                try:
                    available_markets = self.data_manager.get_available_markets()
                except Exception:
                    pass
            
            return {
                'fallback_generation_available': data_available,
                'available_markets': available_markets,
                'supported_features': {
                    'basic_plan_generation': data_available,
                    'format_selection': data_available,
                    'budget_allocation': data_available,
                    'plan_comparison': True,  # Always available
                    'input_validation': True  # Always available
                },
                'limitations': {
                    'no_ai_optimization': True,
                    'basic_site_recommendations': True,
                    'limited_strategic_insights': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking offline capabilities: {str(e)}")
            return {
                'fallback_generation_available': False,
                'error': str(e)
            }

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status and health information.
        
        Returns:
            Dictionary with system status information
        """
        try:
            # Data status
            data_status = self.data_manager.validate_data_freshness()
            
            # Available markets
            available_markets = self.data_manager.get_available_markets()
            
            # AI model info
            model_info = self.ai_generator.get_model_info()
            
            # Cache stats
            cache_stats = self.data_manager.get_cache_stats()
            
            return {
                'data_status': data_status,
                'available_markets': available_markets,
                'market_count': len(available_markets),
                'ai_model': model_info,
                'cache_stats': cache_stats,
                'system_ready': data_status['overall_status'] in ['ready', 'needs_refresh'],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'error': str(e),
                'system_ready': False,
                'last_updated': datetime.now().isoformat()
            }