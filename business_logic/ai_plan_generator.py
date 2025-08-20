"""
AI Plan Generator for creating intelligent media plans using OpenAI.

This module handles the integration with OpenAI's API to generate strategic
media plans based on client briefs, market data, and optimization constraints.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import openai
from openai import OpenAI

from models.data_models import ClientBrief, MediaPlan, FormatAllocation
from config.settings import config_manager
from data.manager import DataManager
from .error_handler import error_handler, RetryConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIPlanGenerator:
    """
    Generates AI-powered media plans using OpenAI's GPT models.
    
    Handles prompt engineering, API integration, plan generation,
    and optimization for strategic diversity and budget efficiency.
    """
    
    def __init__(self, data_manager: Optional[DataManager] = None, skip_openai_init: bool = False):
        """
        Initialize the AI Plan Generator.
        
        Args:
            data_manager: DataManager instance for accessing market data
            skip_openai_init: Skip OpenAI client initialization (for testing)
        """
        self.data_manager = data_manager or DataManager()
        self.client = None
        if not skip_openai_init:
            self._initialize_openai_client()
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.model_name = "gpt-4"
        self.fine_tuned_model = None
        self.temperature = 0.7
        self.max_tokens = 2000
        
        # Model selection and performance tracking
        self.model_selection_strategy = "auto"  # auto, base_only, fine_tuned_only
        self.cost_tracker = {
            'base_model_calls': 0,
            'fine_tuned_model_calls': 0,
            'base_model_tokens': 0,
            'fine_tuned_model_tokens': 0,
            'total_cost': 0.0
        }
        self.performance_metrics = {
            'base_model': {'response_times': [], 'success_rate': 0.0, 'quality_scores': []},
            'fine_tuned_model': {'response_times': [], 'success_rate': 0.0, 'quality_scores': []}
        }
        
        # Plan generation settings
        self.plan_count = 3
        self.diversity_threshold = 0.3  # Minimum difference between plans
        
    def _initialize_openai_client(self):
        """Initialize OpenAI client with API key."""
        try:
            api_key = config_manager.get_openai_api_key()
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    def create_system_prompt(self, market_data: Dict[str, Any], client_brief: ClientBrief) -> str:
        """
        Create dynamic system prompt based on market data and client constraints.
        
        Args:
            market_data: Market-specific rate card and site data
            client_brief: Client campaign requirements
            
        Returns:
            Formatted system prompt string
        """
        try:
            # Extract available formats and rates
            impact_formats = market_data.get('rate_card', {}).get('impact_formats', {})
            reach_formats = market_data.get('rate_card', {}).get('reach_formats', {})
            sites_info = market_data.get('sites', {})
            
            # Build format information
            format_details = []
            
            # Add Impact formats
            for format_name, rate_info in impact_formats.items():
                cpm = rate_info.get('cpm', 'N/A')
                sites = sites_info.get('sites_by_format', {}).get(format_name, [])
                site_count = len(sites)
                
                format_details.append(
                    f"- {format_name} (APX Impact): CPM ${cpm}, {site_count} sites available"
                )
            
            # Add Reach formats
            for format_name, rate_info in reach_formats.items():
                cpm = rate_info.get('cpm', 'N/A')
                sites = sites_info.get('sites_by_format', {}).get(format_name, [])
                site_count = len(sites)
                
                format_details.append(
                    f"- {format_name} (Reach Media): CPM ${cpm}, {site_count} sites available"
                )
            
            # Create constraint information
            constraints = []
            if client_brief.selected_formats:
                constraints.append(f"Must use only these formats: {', '.join(client_brief.selected_formats)}")
            
            # Build the system prompt
            system_prompt = f"""You are an expert media planner at Adzymic, a leading digital advertising agency. Your task is to create strategic media plans that optimize reach, frequency, and budget efficiency for clients.

CLIENT BRIEF:
- Brand: {client_brief.brand_name}
- Budget: ${client_brief.budget:,.2f}
- Market: {client_brief.country}
- Campaign Period: {client_brief.campaign_period}
- Objective: {client_brief.objective}
- Planning Mode: {client_brief.planning_mode}

AVAILABLE AD FORMATS IN {client_brief.country}:
{chr(10).join(format_details)}

CONSTRAINTS:
- Total budget must not exceed ${client_brief.budget:,.2f}
- All allocations must use available formats and realistic CPM rates
{chr(10).join(f"- {constraint}" for constraint in constraints) if constraints else ""}

OPTIMIZATION PRINCIPLES:
1. Reach Optimization: Prioritize formats that maximize unique audience reach
2. Frequency Management: Balance reach vs frequency based on campaign objectives
3. Budget Efficiency: Allocate budget to formats with best cost-per-impression ratios
4. Strategic Diversity: Create meaningfully different approaches across multiple plans
5. Market Relevance: Consider local market preferences and site popularity

PLAN REQUIREMENTS:
- Generate exactly 3 distinct strategic approaches
- Each plan should have a clear strategic rationale
- Provide specific budget allocations per format
- Include estimated impressions and reach where calculable
- Ensure plans are strategically different (not just minor budget variations)

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
    "plans": [
        {{
            "title": "Strategic Plan Name",
            "rationale": "Clear explanation of the strategic approach and why it's optimal",
            "total_budget": budget_number,
            "allocations": [
                {{
                    "format_name": "Exact format name from available list",
                    "budget_allocation": allocation_amount,
                    "cpm": cpm_rate,
                    "estimated_impressions": calculated_impressions,
                    "recommended_sites": ["site1", "site2"],
                    "notes": "Specific reasoning for this allocation"
                }}
            ],
            "estimated_reach": estimated_unique_reach,
            "estimated_impressions": total_impressions_across_formats
        }}
    ]
}}

Focus on creating strategically distinct plans that offer genuine alternatives, not just budget variations."""

            return system_prompt
            
        except Exception as e:
            logger.error(f"Error creating system prompt: {str(e)}")
            raise
    
    def generate_multiple_plans(self, client_brief: ClientBrief, count: int = 3) -> List[MediaPlan]:
        """
        Generate multiple distinct media plan options with comprehensive error handling.
        
        Args:
            client_brief: Client campaign requirements
            count: Number of plans to generate (default 3)
            
        Returns:
            List of MediaPlan objects
            
        Raises:
            ValueError: If unable to generate valid plans
            Exception: For API or processing errors
        """
        try:
            # Get market data with error handling
            def get_market_data():
                return self.data_manager.get_market_data(client_brief.country)
            
            success, market_data, error_info = error_handler.retry_with_backoff(
                get_market_data,
                RetryConfig(max_attempts=2, base_delay=1.0),
                "market data retrieval"
            )
            
            if not success:
                raise Exception(f"Failed to get market data: {error_info.message}")
            
            if not market_data.get('available'):
                raise ValueError(f"Market {client_brief.country} is not available or has no data")
            
            # Create system prompt
            system_prompt = self.create_system_prompt(market_data, client_brief)
            
            # Create user prompt
            user_prompt = f"""Create {count} strategically different media plans for this campaign. 

Each plan should represent a distinct strategic approach:
1. Plan 1: Reach-focused strategy (maximize unique audience)
2. Plan 2: Frequency-focused strategy (deeper engagement with smaller audience)  
3. Plan 3: Balanced/hybrid strategy (optimal reach-frequency balance)

Ensure each plan:
- Uses realistic CPM rates from the available formats
- Stays within the ${client_brief.budget:,.2f} budget
- Provides clear strategic differentiation
- Includes specific site recommendations where possible
- Shows calculated impressions based on budget/CPM

Return the plans in the specified JSON format."""

            # Generate plans with enhanced retry logic
            def call_openai():
                return self._call_openai_api(system_prompt, user_prompt, client_brief)
            
            # Use intelligent retry with rate limit handling
            retry_config = RetryConfig(
                max_attempts=3,
                base_delay=error_handler.get_rate_limit_delay() if hasattr(error_handler, 'rate_limit_tracker') else 2.0,
                exponential_backoff=True,
                max_delay=120.0
            )
            
            success, plans_response, error_info = error_handler.retry_with_backoff(
                call_openai,
                retry_config,
                "OpenAI plan generation"
            )
            
            if not success:
                # Log the error and re-raise with context
                error_handler.log_error(error_info, "AI Plan Generation")
                raise Exception(f"AI plan generation failed: {error_info.message}")
            
            # Validate and convert to MediaPlan objects
            validated_plans = self._validate_and_convert_plans(plans_response, client_brief, market_data)
            
            if not validated_plans:
                raise ValueError("AI generated no valid plans that meet the requirements")
            
            # Ensure plan diversity
            diverse_plans = self._ensure_plan_diversity(validated_plans)
            
            # Track plan quality for performance monitoring
            if diverse_plans:
                # Get the model that was used for this generation
                model_used = self.select_optimal_model(client_brief)
                
                # Calculate average quality score for all generated plans
                quality_scores = [self.evaluate_plan_quality(plan, client_brief) for plan in diverse_plans]
                avg_quality = sum(quality_scores) / len(quality_scores)
                
                # Track the quality for the model that was used
                self.track_model_performance(model_used, 0, True, avg_quality)
                
                logger.info(f"Generated plans quality score: {avg_quality:.3f}")
            
            logger.info(f"Successfully generated {len(diverse_plans)} media plans")
            return diverse_plans
            
        except Exception as e:
            # Classify and log the error
            error_info = error_handler.classify_error(e, "plan generation")
            error_handler.log_error(error_info, "AI Plan Generation")
            raise
    
    def _call_openai_api(self, system_prompt: str, user_prompt: str, 
                        client_brief: Optional[ClientBrief] = None) -> Dict[str, Any]:
        """
        Call OpenAI API with comprehensive error handling and performance tracking.
        
        Args:
            system_prompt: System prompt for the AI
            user_prompt: User prompt with specific instructions
            client_brief: Optional client brief for model selection
            
        Returns:
            Parsed JSON response from OpenAI
            
        Raises:
            Exception: For API or processing errors
        """
        start_time = time.time()
        model_used = None
        success = False
        
        try:
            logger.info("Calling OpenAI API for plan generation")
            
            # Validate client is initialized
            if not self.client:
                raise Exception("OpenAI client not initialized. Please check API key configuration.")
            
            # Select optimal model
            if client_brief:
                model = self.select_optimal_model(client_brief)
            else:
                model = self.fine_tuned_model if self.fine_tuned_model else self.model_name
            
            model_used = model
            logger.info(f"Using model: {model}")
            
            # Make API call with timeout
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                timeout=60.0  # 60 second timeout
            )
            
            # Track token usage for cost calculation
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                self.track_model_cost(model, input_tokens, output_tokens)
            
            # Validate response structure
            if not response.choices or len(response.choices) == 0:
                raise Exception("OpenAI returned empty response")
            
            content = response.choices[0].message.content
            if not content:
                raise Exception("OpenAI returned empty content")
            
            # Parse and validate JSON
            try:
                parsed_response = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI JSON response: {content[:500]}...")
                raise Exception(f"Invalid JSON response from OpenAI: {str(e)}")
            
            # Basic structure validation
            if not isinstance(parsed_response, dict):
                raise Exception("OpenAI response is not a valid JSON object")
            
            if 'plans' not in parsed_response:
                raise Exception("OpenAI response missing 'plans' key")
            
            if not isinstance(parsed_response['plans'], list):
                raise Exception("OpenAI response 'plans' is not a list")
            
            if len(parsed_response['plans']) == 0:
                raise Exception("OpenAI returned no plans")
            
            success = True
            logger.info(f"OpenAI API call successful, received {len(parsed_response['plans'])} plans")
            return parsed_response
            
        except openai.RateLimitError as e:
            # Handle rate limiting specifically
            error_handler._track_rate_limit()
            raise e
            
        except openai.AuthenticationError as e:
            raise Exception(f"OpenAI authentication failed: {str(e)}")
            
        except openai.APIConnectionError as e:
            raise Exception(f"Failed to connect to OpenAI: {str(e)}")
            
        except openai.APITimeoutError as e:
            raise Exception(f"OpenAI request timed out: {str(e)}")
            
        except openai.BadRequestError as e:
            raise Exception(f"Invalid request to OpenAI: {str(e)}")
            
        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {str(e)}")
            
        except Exception as e:
            # Re-raise with context if not already handled
            if "OpenAI" not in str(e):
                raise Exception(f"Unexpected error in OpenAI call: {str(e)}")
            raise
            
        finally:
            # Track performance metrics
            if model_used:
                response_time = time.time() - start_time
                self.track_model_performance(model_used, response_time, success)
    
    def _validate_and_convert_plans(self, plans_data: Dict[str, Any], client_brief: ClientBrief, 
                                  market_data: Dict[str, Any]) -> List[MediaPlan]:
        """
        Validate AI-generated plans and convert to MediaPlan objects.
        
        Args:
            plans_data: Raw plans data from OpenAI
            client_brief: Original client brief
            market_data: Market data for validation
            
        Returns:
            List of validated MediaPlan objects
            
        Raises:
            ValueError: If plans are invalid or don't meet requirements
        """
        try:
            if 'plans' not in plans_data:
                raise ValueError("Response missing 'plans' key")
            
            raw_plans = plans_data['plans']
            if not isinstance(raw_plans, list) or len(raw_plans) == 0:
                raise ValueError("No plans found in response")
            
            validated_plans = []
            
            for i, plan_data in enumerate(raw_plans):
                try:
                    # Validate required fields
                    required_fields = ['title', 'rationale', 'total_budget', 'allocations']
                    for field in required_fields:
                        if field not in plan_data:
                            raise ValueError(f"Plan {i+1} missing required field: {field}")
                    
                    # Validate budget constraint
                    total_budget = float(plan_data['total_budget'])
                    if total_budget > client_brief.budget * 1.05:  # Allow 5% tolerance
                        logger.warning(f"Plan {i+1} exceeds budget: ${total_budget:,.2f} > ${client_brief.budget:,.2f}")
                        continue
                    
                    # Validate and convert allocations
                    allocations = []
                    total_allocated = 0
                    
                    for alloc_data in plan_data['allocations']:
                        try:
                            allocation = FormatAllocation(
                                format_name=str(alloc_data['format_name']),
                                budget_allocation=float(alloc_data['budget_allocation']),
                                cpm=float(alloc_data['cpm']),
                                estimated_impressions=int(alloc_data.get('estimated_impressions', 0)),
                                recommended_sites=alloc_data.get('recommended_sites', []),
                                notes=str(alloc_data.get('notes', ''))
                            )
                            
                            # Validate format exists in market data
                            if not self._validate_format_in_market(allocation.format_name, market_data):
                                logger.warning(f"Format {allocation.format_name} not available in market")
                                continue
                            
                            allocations.append(allocation)
                            total_allocated += allocation.budget_allocation
                            
                        except (KeyError, ValueError, TypeError) as e:
                            logger.warning(f"Invalid allocation in plan {i+1}: {str(e)}")
                            continue
                    
                    if not allocations:
                        logger.warning(f"Plan {i+1} has no valid allocations")
                        continue
                    
                    # Create MediaPlan object
                    media_plan = MediaPlan(
                        plan_id=f"plan_{i+1}_{int(time.time())}",
                        title=str(plan_data['title']),
                        total_budget=total_allocated,  # Use actual allocated amount
                        allocations=allocations,
                        estimated_reach=int(plan_data.get('estimated_reach', 0)),
                        estimated_impressions=sum(alloc.estimated_impressions for alloc in allocations),
                        rationale=str(plan_data['rationale']),
                        created_at=datetime.now()
                    )
                    
                    validated_plans.append(media_plan)
                    logger.info(f"Validated plan {i+1}: {media_plan.title}")
                    
                except Exception as e:
                    logger.warning(f"Failed to validate plan {i+1}: {str(e)}")
                    continue
            
            if not validated_plans:
                raise ValueError("No valid plans could be generated from AI response")
            
            return validated_plans
            
        except Exception as e:
            logger.error(f"Error validating and converting plans: {str(e)}")
            raise
    
    def _validate_format_in_market(self, format_name: str, market_data: Dict[str, Any]) -> bool:
        """
        Validate that a format is available in the market.
        
        Args:
            format_name: Name of the ad format
            market_data: Market data dictionary
            
        Returns:
            True if format is available, False otherwise
        """
        try:
            rate_card = market_data.get('rate_card', {})
            impact_formats = rate_card.get('impact_formats', {})
            reach_formats = rate_card.get('reach_formats', {})
            
            return format_name in impact_formats or format_name in reach_formats
            
        except Exception:
            return False
    
    def _ensure_plan_diversity(self, plans: List[MediaPlan]) -> List[MediaPlan]:
        """
        Ensure generated plans have sufficient strategic diversity.
        
        Args:
            plans: List of MediaPlan objects
            
        Returns:
            List of diverse MediaPlan objects
        """
        try:
            if len(plans) <= 1:
                return plans
            
            diverse_plans = [plans[0]]  # Always include first plan
            
            for plan in plans[1:]:
                is_diverse = True
                
                for existing_plan in diverse_plans:
                    similarity = self._calculate_plan_similarity(plan, existing_plan)
                    
                    if similarity > (1.0 - self.diversity_threshold):
                        is_diverse = False
                        break
                
                if is_diverse:
                    diverse_plans.append(plan)
                else:
                    logger.info(f"Filtered out similar plan: {plan.title}")
            
            return diverse_plans
            
        except Exception as e:
            logger.warning(f"Error ensuring plan diversity: {str(e)}")
            return plans  # Return original plans if diversity check fails
    
    def _calculate_plan_similarity(self, plan1: MediaPlan, plan2: MediaPlan) -> float:
        """
        Calculate similarity between two media plans.
        
        Args:
            plan1: First MediaPlan
            plan2: Second MediaPlan
            
        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical)
        """
        try:
            # Compare format allocations
            formats1 = {alloc.format_name: alloc.budget_allocation for alloc in plan1.allocations}
            formats2 = {alloc.format_name: alloc.budget_allocation for alloc in plan2.allocations}
            
            all_formats = set(formats1.keys()) | set(formats2.keys())
            
            if not all_formats:
                return 1.0  # Both plans have no allocations
            
            # Calculate allocation similarity
            allocation_similarity = 0.0
            total_budget = max(plan1.total_budget, plan2.total_budget)
            
            for format_name in all_formats:
                alloc1 = formats1.get(format_name, 0) / total_budget
                alloc2 = formats2.get(format_name, 0) / total_budget
                allocation_similarity += 1.0 - abs(alloc1 - alloc2)
            
            allocation_similarity /= len(all_formats)
            
            # Compare budget similarity
            budget_diff = abs(plan1.total_budget - plan2.total_budget) / max(plan1.total_budget, plan2.total_budget)
            budget_similarity = 1.0 - budget_diff
            
            # Weighted average
            overall_similarity = (allocation_similarity * 0.7) + (budget_similarity * 0.3)
            
            return overall_similarity
            
        except Exception as e:
            logger.warning(f"Error calculating plan similarity: {str(e)}")
            return 0.0  # Assume different if calculation fails
    
    def use_fine_tuned_model(self, model_name: str):
        """
        Switch to using a fine-tuned model.
        
        Args:
            model_name: Name of the fine-tuned model
        """
        try:
            # Validate model exists (basic check)
            if not model_name or not model_name.startswith('ft:'):
                raise ValueError("Invalid fine-tuned model name")
            
            self.fine_tuned_model = model_name
            logger.info(f"Switched to fine-tuned model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error switching to fine-tuned model: {str(e)}")
            raise
    
    def use_base_model(self):
        """Switch back to using the base model."""
        self.fine_tuned_model = None
        logger.info(f"Switched back to base model: {self.model_name}")
    
    def set_model_selection_strategy(self, strategy: str):
        """
        Set the model selection strategy.
        
        Args:
            strategy: One of 'auto', 'base_only', 'fine_tuned_only'
        """
        valid_strategies = ['auto', 'base_only', 'fine_tuned_only']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        
        self.model_selection_strategy = strategy
        logger.info(f"Model selection strategy set to: {strategy}")
    
    def select_optimal_model(self, client_brief: ClientBrief) -> str:
        """
        Select the optimal model based on strategy and performance metrics.
        
        Args:
            client_brief: Client brief to consider for model selection
            
        Returns:
            Model name to use for this request
        """
        try:
            # Strategy-based selection
            if self.model_selection_strategy == "base_only":
                return self.model_name
            
            if self.model_selection_strategy == "fine_tuned_only":
                if self.fine_tuned_model:
                    return self.fine_tuned_model
                else:
                    logger.warning("Fine-tuned model requested but not available, falling back to base model")
                    return self.model_name
            
            # Auto selection based on performance and availability
            if self.model_selection_strategy == "auto":
                return self._auto_select_model(client_brief)
            
            return self.model_name
            
        except Exception as e:
            logger.error(f"Error selecting optimal model: {str(e)}")
            return self.model_name  # Fallback to base model
    
    def _auto_select_model(self, client_brief: ClientBrief) -> str:
        """
        Automatically select the best model based on performance metrics.
        
        Args:
            client_brief: Client brief for context
            
        Returns:
            Selected model name
        """
        try:
            # If no fine-tuned model available, use base model
            if not self.fine_tuned_model:
                return self.model_name
            
            base_metrics = self.performance_metrics['base_model']
            fine_tuned_metrics = self.performance_metrics['fine_tuned_model']
            
            # If we don't have enough data for comparison, use fine-tuned model
            if (len(base_metrics['quality_scores']) < 5 or 
                len(fine_tuned_metrics['quality_scores']) < 5):
                return self.fine_tuned_model
            
            # Compare average quality scores
            base_avg_quality = sum(base_metrics['quality_scores']) / len(base_metrics['quality_scores'])
            fine_tuned_avg_quality = sum(fine_tuned_metrics['quality_scores']) / len(fine_tuned_metrics['quality_scores'])
            
            # Compare success rates
            base_success_rate = base_metrics['success_rate']
            fine_tuned_success_rate = fine_tuned_metrics['success_rate']
            
            # Compare average response times
            base_avg_time = sum(base_metrics['response_times']) / len(base_metrics['response_times']) if base_metrics['response_times'] else 0
            fine_tuned_avg_time = sum(fine_tuned_metrics['response_times']) / len(fine_tuned_metrics['response_times']) if fine_tuned_metrics['response_times'] else 0
            
            # Decision logic: prioritize quality and success rate over speed
            quality_threshold = 0.05  # 5% improvement threshold
            
            if (fine_tuned_avg_quality > base_avg_quality + quality_threshold and 
                fine_tuned_success_rate >= base_success_rate - 0.05):
                logger.info(f"Selected fine-tuned model based on quality: {fine_tuned_avg_quality:.3f} vs {base_avg_quality:.3f}")
                return self.fine_tuned_model
            
            elif (base_avg_quality > fine_tuned_avg_quality + quality_threshold and 
                  base_success_rate >= fine_tuned_success_rate - 0.05):
                logger.info(f"Selected base model based on quality: {base_avg_quality:.3f} vs {fine_tuned_avg_quality:.3f}")
                return self.model_name
            
            # If quality is similar, consider cost (fine-tuned models are typically more expensive)
            # For now, prefer fine-tuned model if quality is similar (assuming it's been trained for this domain)
            logger.info("Quality metrics similar, preferring fine-tuned model for domain specialization")
            return self.fine_tuned_model
            
        except Exception as e:
            logger.error(f"Error in auto model selection: {str(e)}")
            return self.model_name  # Fallback to base model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently configured model.
        
        Returns:
            Dictionary with model configuration details
        """
        return {
            'base_model': self.model_name,
            'fine_tuned_model': self.fine_tuned_model,
            'active_model': self.fine_tuned_model or self.model_name,
            'selection_strategy': self.model_selection_strategy,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'max_retries': self.max_retries,
            'cost_tracker': self.cost_tracker.copy(),
            'performance_summary': self._get_performance_summary()
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics for both models."""
        try:
            summary = {}
            
            for model_type in ['base_model', 'fine_tuned_model']:
                metrics = self.performance_metrics[model_type]
                
                if metrics['response_times']:
                    avg_response_time = sum(metrics['response_times']) / len(metrics['response_times'])
                    min_response_time = min(metrics['response_times'])
                    max_response_time = max(metrics['response_times'])
                else:
                    avg_response_time = min_response_time = max_response_time = 0
                
                if metrics['quality_scores']:
                    avg_quality = sum(metrics['quality_scores']) / len(metrics['quality_scores'])
                    min_quality = min(metrics['quality_scores'])
                    max_quality = max(metrics['quality_scores'])
                else:
                    avg_quality = min_quality = max_quality = 0
                
                summary[model_type] = {
                    'total_calls': len(metrics['response_times']),
                    'success_rate': metrics['success_rate'],
                    'avg_response_time': avg_response_time,
                    'min_response_time': min_response_time,
                    'max_response_time': max_response_time,
                    'avg_quality_score': avg_quality,
                    'min_quality_score': min_quality,
                    'max_quality_score': max_quality
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return {}
    
    def track_model_performance(self, model_used: str, response_time: float, 
                              success: bool, quality_score: Optional[float] = None):
        """
        Track performance metrics for model usage.
        
        Args:
            model_used: Name of the model that was used
            response_time: Response time in seconds
            success: Whether the request was successful
            quality_score: Optional quality score (0.0 to 1.0)
        """
        try:
            # Determine model type
            model_type = 'fine_tuned_model' if model_used.startswith('ft:') else 'base_model'
            
            metrics = self.performance_metrics[model_type]
            
            # Track response time
            metrics['response_times'].append(response_time)
            
            # Keep only recent metrics (last 100 calls)
            if len(metrics['response_times']) > 100:
                metrics['response_times'] = metrics['response_times'][-100:]
            
            # Update success rate
            total_calls = len(metrics['response_times'])
            if total_calls > 0:
                # Calculate success rate from recent history
                recent_successes = getattr(self, f'_{model_type}_recent_successes', [])
                recent_successes.append(success)
                
                if len(recent_successes) > 100:
                    recent_successes = recent_successes[-100:]
                
                setattr(self, f'_{model_type}_recent_successes', recent_successes)
                metrics['success_rate'] = sum(recent_successes) / len(recent_successes)
            
            # Track quality score if provided
            if quality_score is not None:
                metrics['quality_scores'].append(quality_score)
                
                # Keep only recent quality scores
                if len(metrics['quality_scores']) > 100:
                    metrics['quality_scores'] = metrics['quality_scores'][-100:]
            
            logger.debug(f"Tracked performance for {model_type}: time={response_time:.2f}s, success={success}, quality={quality_score}")
            
        except Exception as e:
            logger.error(f"Error tracking model performance: {str(e)}")
    
    def track_model_cost(self, model_used: str, input_tokens: int, output_tokens: int):
        """
        Track cost metrics for model usage.
        
        Args:
            model_used: Name of the model that was used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        try:
            # Determine model type and update counters
            if model_used.startswith('ft:'):
                self.cost_tracker['fine_tuned_model_calls'] += 1
                self.cost_tracker['fine_tuned_model_tokens'] += input_tokens + output_tokens
                
                # Fine-tuned model pricing (approximate - actual pricing may vary)
                # These are example rates and should be updated based on actual OpenAI pricing
                input_cost = input_tokens * 0.012 / 1000  # $0.012 per 1K input tokens
                output_cost = output_tokens * 0.016 / 1000  # $0.016 per 1K output tokens
                
            else:
                self.cost_tracker['base_model_calls'] += 1
                self.cost_tracker['base_model_tokens'] += input_tokens + output_tokens
                
                # Base model pricing (GPT-4 example rates)
                input_cost = input_tokens * 0.03 / 1000  # $0.03 per 1K input tokens
                output_cost = output_tokens * 0.06 / 1000  # $0.06 per 1K output tokens
            
            call_cost = input_cost + output_cost
            self.cost_tracker['total_cost'] += call_cost
            
            logger.debug(f"Tracked cost for {model_used}: ${call_cost:.4f} ({input_tokens} in + {output_tokens} out tokens)")
            
        except Exception as e:
            logger.error(f"Error tracking model cost: {str(e)}")
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """
        Get detailed cost analysis for model usage.
        
        Returns:
            Dictionary with cost breakdown and recommendations
        """
        try:
            tracker = self.cost_tracker
            
            # Calculate costs per call
            base_cost_per_call = 0
            fine_tuned_cost_per_call = 0
            
            if tracker['base_model_calls'] > 0:
                base_total_cost = (tracker['base_model_tokens'] * 0.045 / 1000)  # Average of input/output
                base_cost_per_call = base_total_cost / tracker['base_model_calls']
            
            if tracker['fine_tuned_model_calls'] > 0:
                fine_tuned_total_cost = (tracker['fine_tuned_model_tokens'] * 0.014 / 1000)  # Average of input/output
                fine_tuned_cost_per_call = fine_tuned_total_cost / tracker['fine_tuned_model_calls']
            
            # Generate recommendations
            recommendations = []
            
            if tracker['fine_tuned_model_calls'] > 0 and tracker['base_model_calls'] > 0:
                if fine_tuned_cost_per_call < base_cost_per_call:
                    savings_per_call = base_cost_per_call - fine_tuned_cost_per_call
                    recommendations.append(f"Fine-tuned model is ${savings_per_call:.4f} cheaper per call")
                else:
                    extra_cost_per_call = fine_tuned_cost_per_call - base_cost_per_call
                    recommendations.append(f"Fine-tuned model costs ${extra_cost_per_call:.4f} more per call")
            
            if tracker['total_cost'] > 100:  # If spending more than $100
                recommendations.append("Consider optimizing prompts to reduce token usage")
            
            return {
                'total_cost': tracker['total_cost'],
                'base_model': {
                    'calls': tracker['base_model_calls'],
                    'tokens': tracker['base_model_tokens'],
                    'cost_per_call': base_cost_per_call,
                    'estimated_total_cost': base_cost_per_call * tracker['base_model_calls']
                },
                'fine_tuned_model': {
                    'calls': tracker['fine_tuned_model_calls'],
                    'tokens': tracker['fine_tuned_model_tokens'],
                    'cost_per_call': fine_tuned_cost_per_call,
                    'estimated_total_cost': fine_tuned_cost_per_call * tracker['fine_tuned_model_calls']
                },
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating cost analysis: {str(e)}")
            return {'error': str(e)}
    
    def reset_performance_metrics(self):
        """Reset all performance and cost tracking metrics."""
        try:
            self.cost_tracker = {
                'base_model_calls': 0,
                'fine_tuned_model_calls': 0,
                'base_model_tokens': 0,
                'fine_tuned_model_tokens': 0,
                'total_cost': 0.0
            }
            
            self.performance_metrics = {
                'base_model': {'response_times': [], 'success_rate': 0.0, 'quality_scores': []},
                'fine_tuned_model': {'response_times': [], 'success_rate': 0.0, 'quality_scores': []}
            }
            
            # Reset success tracking
            self._base_model_recent_successes = []
            self._fine_tuned_model_recent_successes = []
            
            logger.info("Performance and cost metrics reset")
            
        except Exception as e:
            logger.error(f"Error resetting performance metrics: {str(e)}")
    
    def evaluate_plan_quality(self, media_plan: MediaPlan, client_brief: ClientBrief) -> float:
        """
        Evaluate the quality of a generated media plan.
        
        Args:
            media_plan: The generated media plan
            client_brief: The original client brief
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            quality_score = 0.0
            max_score = 0.0
            
            # Budget adherence (25% of score)
            budget_adherence = min(1.0, client_brief.budget / max(media_plan.total_budget, 1))
            if media_plan.total_budget <= client_brief.budget:
                budget_score = 1.0
            else:
                budget_score = max(0.0, 1.0 - (media_plan.total_budget - client_brief.budget) / client_brief.budget)
            
            quality_score += budget_score * 0.25
            max_score += 0.25
            
            # Allocation diversity (20% of score)
            if media_plan.allocations:
                unique_formats = len(set(alloc.format_name for alloc in media_plan.allocations))
                diversity_score = min(1.0, unique_formats / 3)  # Normalize to max 3 formats
                quality_score += diversity_score * 0.20
            max_score += 0.20
            
            # Rationale quality (15% of score)
            if media_plan.rationale and len(media_plan.rationale.strip()) > 50:
                rationale_score = min(1.0, len(media_plan.rationale.strip()) / 200)  # Normalize to 200 chars
                quality_score += rationale_score * 0.15
            max_score += 0.15
            
            # Allocation completeness (20% of score)
            complete_allocations = 0
            for alloc in media_plan.allocations:
                if (alloc.budget_allocation > 0 and alloc.cpm > 0 and 
                    alloc.estimated_impressions > 0 and alloc.recommended_sites):
                    complete_allocations += 1
            
            if media_plan.allocations:
                completeness_score = complete_allocations / len(media_plan.allocations)
                quality_score += completeness_score * 0.20
            max_score += 0.20
            
            # Reach and impression estimates (20% of score)
            if media_plan.estimated_reach > 0 and media_plan.estimated_impressions > 0:
                # Basic sanity check: reach should be less than impressions
                if media_plan.estimated_reach <= media_plan.estimated_impressions:
                    estimates_score = 1.0
                else:
                    estimates_score = 0.5  # Partial credit for having estimates
                quality_score += estimates_score * 0.20
            max_score += 0.20
            
            # Normalize to 0-1 scale
            final_score = quality_score / max_score if max_score > 0 else 0.0
            
            logger.debug(f"Plan quality evaluation: {final_score:.3f} (budget: {budget_score:.2f}, diversity: {diversity_score:.2f})")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error evaluating plan quality: {str(e)}")
            return 0.0
    
    def optimize_plan_diversity(self, plans: List[MediaPlan], target_count: int = 3) -> List[MediaPlan]:
        """
        Optimize plan diversity by selecting the most strategically different plans.
        
        Args:
            plans: List of MediaPlan objects to optimize
            target_count: Target number of diverse plans to return
            
        Returns:
            List of optimally diverse MediaPlan objects
        """
        try:
            if len(plans) <= target_count:
                return plans
            
            # Start with the first plan
            selected_plans = [plans[0]]
            remaining_plans = plans[1:]
            
            # Greedily select most diverse plans
            while len(selected_plans) < target_count and remaining_plans:
                best_plan = None
                best_diversity_score = -1
                
                for candidate in remaining_plans:
                    # Calculate minimum diversity to all selected plans
                    min_diversity = min(
                        1.0 - self._calculate_plan_similarity(candidate, selected)
                        for selected in selected_plans
                    )
                    
                    if min_diversity > best_diversity_score:
                        best_diversity_score = min_diversity
                        best_plan = candidate
                
                if best_plan and best_diversity_score >= self.diversity_threshold:
                    selected_plans.append(best_plan)
                    remaining_plans.remove(best_plan)
                else:
                    break  # No more diverse plans available
            
            logger.info(f"Optimized to {len(selected_plans)} diverse plans from {len(plans)} total")
            return selected_plans
            
        except Exception as e:
            logger.error(f"Error optimizing plan diversity: {str(e)}")
            return plans[:target_count]  # Fallback to first N plans