"""
Budget optimization and allocation logic for media planning.

This module provides algorithms for optimal budget distribution across formats,
reach optimization, frequency management, and strategic allocation decisions.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from models.data_models import ClientBrief, FormatAllocation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    REACH_FOCUSED = "reach_focused"
    FREQUENCY_FOCUSED = "frequency_focused"
    BALANCED = "balanced"
    COST_EFFICIENT = "cost_efficient"
    HIGH_IMPACT = "high_impact"


@dataclass
class FormatMetrics:
    """Metrics for evaluating ad format performance."""
    format_name: str
    cpm: float
    estimated_reach_per_1k: int  # Estimated unique reach per 1000 impressions
    frequency_cap: float  # Optimal frequency for this format
    impact_score: float  # Relative impact/quality score (1-10)
    site_count: int  # Number of available sites
    market_share: float  # Format's share of market inventory


@dataclass
class AllocationResult:
    """Result of budget allocation optimization."""
    allocations: List[FormatAllocation]
    total_budget_used: float
    estimated_total_reach: int
    estimated_total_impressions: int
    average_frequency: float
    optimization_score: float
    strategy_notes: str


class BudgetOptimizer:
    """
    Handles budget optimization and allocation across ad formats.
    
    Implements various optimization strategies including reach maximization,
    frequency optimization, and cost efficiency algorithms.
    """
    
    def __init__(self):
        """Initialize the budget optimizer."""
        # Optimization parameters
        self.min_allocation_percentage = 0.05  # Minimum 5% allocation per format
        self.max_allocation_percentage = 0.70  # Maximum 70% allocation per format
        self.frequency_sweet_spot = 3.0  # Optimal frequency for most campaigns
        self.reach_decay_factor = 0.85  # Diminishing returns factor for reach
        
        # Format performance assumptions (can be overridden with real data)
        self.default_reach_efficiency = {
            'display': 0.65,
            'video': 0.45,
            'native': 0.70,
            'mobile': 0.60,
            'social': 0.55
        }
        
        self.default_impact_scores = {
            'display': 6.0,
            'video': 8.5,
            'native': 7.0,
            'mobile': 6.5,
            'social': 7.5
        }
    
    def optimize_budget_allocation(self, 
                                 client_brief: ClientBrief,
                                 available_formats: Dict[str, Dict[str, Any]],
                                 strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                                 format_constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> AllocationResult:
        """
        Optimize budget allocation across available formats.
        
        Args:
            client_brief: Client campaign requirements
            available_formats: Dictionary of format data with CPM and site info
            strategy: Optimization strategy to use
            format_constraints: Optional min/max allocation constraints per format
            
        Returns:
            AllocationResult with optimized allocations
        """
        try:
            logger.info(f"Optimizing budget allocation with {strategy.value} strategy")
            
            # Filter formats based on client selection
            if client_brief.selected_formats:
                filtered_formats = {
                    name: data for name, data in available_formats.items()
                    if name in client_brief.selected_formats
                }
            else:
                filtered_formats = available_formats
            
            if not filtered_formats:
                raise ValueError("No available formats for optimization")
            
            # Calculate format metrics
            format_metrics = self._calculate_format_metrics(filtered_formats)
            
            # Apply optimization strategy
            if strategy == OptimizationStrategy.REACH_FOCUSED:
                allocations = self._optimize_for_reach(client_brief.budget, format_metrics)
            elif strategy == OptimizationStrategy.FREQUENCY_FOCUSED:
                allocations = self._optimize_for_frequency(client_brief.budget, format_metrics)
            elif strategy == OptimizationStrategy.COST_EFFICIENT:
                allocations = self._optimize_for_cost_efficiency(client_brief.budget, format_metrics)
            elif strategy == OptimizationStrategy.HIGH_IMPACT:
                allocations = self._optimize_for_high_impact(client_brief.budget, format_metrics)
            else:  # BALANCED
                allocations = self._optimize_balanced(client_brief.budget, format_metrics)
            
            # Apply constraints if provided
            if format_constraints:
                allocations = self._apply_allocation_constraints(allocations, format_constraints, client_brief.budget)
            
            # Calculate performance metrics
            result = self._calculate_allocation_metrics(allocations, format_metrics, strategy)
            
            logger.info(f"Optimization complete: ${result.total_budget_used:,.2f} allocated across {len(result.allocations)} formats")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing budget allocation: {str(e)}")
            raise
    
    def _calculate_format_metrics(self, available_formats: Dict[str, Dict[str, Any]]) -> List[FormatMetrics]:
        """
        Calculate performance metrics for each available format.
        
        Args:
            available_formats: Dictionary of format data
            
        Returns:
            List of FormatMetrics objects
        """
        try:
            metrics = []
            
            for format_name, format_data in available_formats.items():
                # Extract CPM
                cpm = format_data.get('cpm', 0)
                if cpm <= 0:
                    logger.warning(f"Invalid CPM for format {format_name}: {cpm}")
                    continue
                
                # Estimate reach efficiency (unique reach per 1000 impressions)
                format_type = self._categorize_format(format_name)
                reach_efficiency = self.default_reach_efficiency.get(format_type, 0.60)
                estimated_reach_per_1k = int(1000 * reach_efficiency)
                
                # Set frequency cap based on format type
                frequency_cap = self._get_optimal_frequency(format_type)
                
                # Get impact score
                impact_score = self.default_impact_scores.get(format_type, 7.0)
                
                # Site count
                site_count = len(format_data.get('sites', []))
                
                # Market share (simplified calculation)
                market_share = min(site_count / 100.0, 1.0)  # Assume 100 sites = 100% share
                
                metrics.append(FormatMetrics(
                    format_name=format_name,
                    cpm=cpm,
                    estimated_reach_per_1k=estimated_reach_per_1k,
                    frequency_cap=frequency_cap,
                    impact_score=impact_score,
                    site_count=site_count,
                    market_share=market_share
                ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating format metrics: {str(e)}")
            raise
    
    def _categorize_format(self, format_name: str) -> str:
        """
        Categorize format name into standard types.
        
        Args:
            format_name: Name of the ad format
            
        Returns:
            Standardized format category
        """
        format_lower = format_name.lower()
        
        if any(keyword in format_lower for keyword in ['video', 'pre-roll', 'mid-roll']):
            return 'video'
        elif any(keyword in format_lower for keyword in ['native', 'content', 'sponsored']):
            return 'native'
        elif any(keyword in format_lower for keyword in ['mobile', 'app', 'in-app']):
            return 'mobile'
        elif any(keyword in format_lower for keyword in ['social', 'facebook', 'instagram', 'twitter']):
            return 'social'
        else:
            return 'display'  # Default category
    
    def _get_optimal_frequency(self, format_type: str) -> float:
        """
        Get optimal frequency for a format type.
        
        Args:
            format_type: Standardized format category
            
        Returns:
            Optimal frequency value
        """
        frequency_map = {
            'video': 2.5,      # Lower frequency for high-impact video
            'display': 4.0,    # Higher frequency for display
            'native': 3.0,     # Moderate frequency for native
            'mobile': 3.5,     # Slightly higher for mobile
            'social': 2.8      # Lower frequency for social
        }
        
        return frequency_map.get(format_type, self.frequency_sweet_spot)
    
    def _optimize_for_reach(self, budget: float, format_metrics: List[FormatMetrics]) -> List[FormatAllocation]:
        """
        Optimize allocation to maximize unique reach.
        
        Args:
            budget: Total available budget
            format_metrics: List of format performance metrics
            
        Returns:
            List of optimized FormatAllocation objects
        """
        try:
            # Sort formats by reach efficiency (reach per dollar)
            sorted_formats = sorted(
                format_metrics,
                key=lambda f: f.estimated_reach_per_1k / f.cpm,
                reverse=True
            )
            
            allocations = []
            remaining_budget = budget
            
            # Allocate budget prioritizing reach efficiency
            for i, format_metric in enumerate(sorted_formats):
                if remaining_budget <= 0:
                    break
                
                # Calculate allocation based on reach efficiency and remaining budget
                if i == 0:  # Give primary format larger share
                    allocation_pct = min(0.50, remaining_budget / budget)
                elif i == 1:  # Secondary format
                    allocation_pct = min(0.35, remaining_budget / budget)
                else:  # Remaining formats
                    allocation_pct = min(0.15, remaining_budget / budget)
                
                allocation_amount = budget * allocation_pct
                
                # Ensure minimum allocation
                min_allocation = budget * self.min_allocation_percentage
                allocation_amount = max(allocation_amount, min_allocation)
                allocation_amount = min(allocation_amount, remaining_budget)
                
                if allocation_amount > 0:
                    impressions = int((allocation_amount / format_metric.cpm) * 1000)
                    
                    allocation = FormatAllocation(
                        format_name=format_metric.format_name,
                        budget_allocation=allocation_amount,
                        cpm=format_metric.cpm,
                        estimated_impressions=impressions,
                        recommended_sites=[],  # Will be populated later
                        notes=f"Reach-optimized allocation (reach efficiency: {format_metric.estimated_reach_per_1k / format_metric.cpm:.1f})"
                    )
                    
                    allocations.append(allocation)
                    remaining_budget -= allocation_amount
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error optimizing for reach: {str(e)}")
            raise
    
    def _optimize_for_frequency(self, budget: float, format_metrics: List[FormatMetrics]) -> List[FormatAllocation]:
        """
        Optimize allocation for deeper engagement through higher frequency.
        
        Args:
            budget: Total available budget
            format_metrics: List of format performance metrics
            
        Returns:
            List of optimized FormatAllocation objects
        """
        try:
            # Sort formats by impact score and frequency suitability
            sorted_formats = sorted(
                format_metrics,
                key=lambda f: f.impact_score * (1 / f.frequency_cap),  # Higher impact, lower frequency cap = better
                reverse=True
            )
            
            allocations = []
            remaining_budget = budget
            
            # Focus budget on fewer, high-impact formats
            selected_formats = sorted_formats[:min(2, len(sorted_formats))]  # Max 2 formats for frequency focus
            
            for i, format_metric in enumerate(selected_formats):
                if remaining_budget <= 0:
                    break
                
                # Allocate larger shares to fewer formats
                if i == 0:
                    allocation_pct = 0.70  # 70% to primary format
                else:
                    allocation_pct = 0.30  # 30% to secondary format
                
                allocation_amount = budget * allocation_pct
                allocation_amount = min(allocation_amount, remaining_budget)
                
                if allocation_amount > 0:
                    impressions = int((allocation_amount / format_metric.cpm) * 1000)
                    
                    allocation = FormatAllocation(
                        format_name=format_metric.format_name,
                        budget_allocation=allocation_amount,
                        cpm=format_metric.cpm,
                        estimated_impressions=impressions,
                        recommended_sites=[],
                        notes=f"Frequency-focused allocation (target frequency: {format_metric.frequency_cap:.1f}x)"
                    )
                    
                    allocations.append(allocation)
                    remaining_budget -= allocation_amount
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error optimizing for frequency: {str(e)}")
            raise
    
    def _optimize_for_cost_efficiency(self, budget: float, format_metrics: List[FormatMetrics]) -> List[FormatAllocation]:
        """
        Optimize allocation for maximum cost efficiency (lowest CPM).
        
        Args:
            budget: Total available budget
            format_metrics: List of format performance metrics
            
        Returns:
            List of optimized FormatAllocation objects
        """
        try:
            # Sort formats by CPM (lowest first)
            sorted_formats = sorted(format_metrics, key=lambda f: f.cpm)
            
            allocations = []
            remaining_budget = budget
            
            # Distribute budget based on cost efficiency
            total_efficiency_score = sum(1 / f.cpm for f in sorted_formats)
            
            for format_metric in sorted_formats:
                if remaining_budget <= 0:
                    break
                
                # Allocate based on inverse CPM (lower CPM = higher allocation)
                efficiency_score = 1 / format_metric.cpm
                allocation_pct = efficiency_score / total_efficiency_score
                
                # Apply min/max constraints
                allocation_pct = max(allocation_pct, self.min_allocation_percentage)
                allocation_pct = min(allocation_pct, self.max_allocation_percentage)
                
                allocation_amount = budget * allocation_pct
                allocation_amount = min(allocation_amount, remaining_budget)
                
                if allocation_amount > 0:
                    impressions = int((allocation_amount / format_metric.cpm) * 1000)
                    
                    allocation = FormatAllocation(
                        format_name=format_metric.format_name,
                        budget_allocation=allocation_amount,
                        cpm=format_metric.cpm,
                        estimated_impressions=impressions,
                        recommended_sites=[],
                        notes=f"Cost-efficient allocation (CPM: ${format_metric.cpm:.2f})"
                    )
                    
                    allocations.append(allocation)
                    remaining_budget -= allocation_amount
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error optimizing for cost efficiency: {str(e)}")
            raise
    
    def _optimize_for_high_impact(self, budget: float, format_metrics: List[FormatMetrics]) -> List[FormatAllocation]:
        """
        Optimize allocation for maximum impact and quality.
        
        Args:
            budget: Total available budget
            format_metrics: List of format performance metrics
            
        Returns:
            List of optimized FormatAllocation objects
        """
        try:
            # Sort formats by impact score
            sorted_formats = sorted(format_metrics, key=lambda f: f.impact_score, reverse=True)
            
            allocations = []
            remaining_budget = budget
            
            # Focus on high-impact formats with premium pricing tolerance
            for i, format_metric in enumerate(sorted_formats):
                if remaining_budget <= 0:
                    break
                
                # Weight allocation by impact score
                if format_metric.impact_score >= 8.0:  # Premium formats
                    allocation_pct = 0.40
                elif format_metric.impact_score >= 7.0:  # High-quality formats
                    allocation_pct = 0.30
                else:  # Standard formats
                    allocation_pct = 0.15
                
                # Adjust for position in list
                allocation_pct *= (1.0 - (i * 0.1))  # Reduce by 10% for each position
                allocation_pct = max(allocation_pct, self.min_allocation_percentage)
                
                allocation_amount = budget * allocation_pct
                allocation_amount = min(allocation_amount, remaining_budget)
                
                if allocation_amount > 0:
                    impressions = int((allocation_amount / format_metric.cpm) * 1000)
                    
                    allocation = FormatAllocation(
                        format_name=format_metric.format_name,
                        budget_allocation=allocation_amount,
                        cpm=format_metric.cpm,
                        estimated_impressions=impressions,
                        recommended_sites=[],
                        notes=f"High-impact allocation (impact score: {format_metric.impact_score:.1f}/10)"
                    )
                    
                    allocations.append(allocation)
                    remaining_budget -= allocation_amount
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error optimizing for high impact: {str(e)}")
            raise
    
    def _optimize_balanced(self, budget: float, format_metrics: List[FormatMetrics]) -> List[FormatAllocation]:
        """
        Optimize allocation using a balanced approach considering multiple factors.
        
        Args:
            budget: Total available budget
            format_metrics: List of format performance metrics
            
        Returns:
            List of optimized FormatAllocation objects
        """
        try:
            # Calculate composite score for each format
            scored_formats = []
            
            for format_metric in format_metrics:
                # Normalize metrics (0-1 scale)
                reach_efficiency = format_metric.estimated_reach_per_1k / format_metric.cpm
                cost_efficiency = 1 / format_metric.cpm  # Inverse CPM
                impact_normalized = format_metric.impact_score / 10.0
                market_share_normalized = format_metric.market_share
                
                # Weighted composite score
                composite_score = (
                    reach_efficiency * 0.30 +      # 30% reach efficiency
                    cost_efficiency * 0.25 +       # 25% cost efficiency  
                    impact_normalized * 0.25 +     # 25% impact quality
                    market_share_normalized * 0.20 # 20% market availability
                )
                
                scored_formats.append((format_metric, composite_score))
            
            # Sort by composite score
            scored_formats.sort(key=lambda x: x[1], reverse=True)
            
            allocations = []
            remaining_budget = budget
            
            # Distribute budget based on composite scores
            total_score = sum(score for _, score in scored_formats)
            
            for format_metric, score in scored_formats:
                if remaining_budget <= 0:
                    break
                
                # Base allocation on score proportion
                allocation_pct = score / total_score
                
                # Apply constraints
                allocation_pct = max(allocation_pct, self.min_allocation_percentage)
                allocation_pct = min(allocation_pct, self.max_allocation_percentage)
                
                allocation_amount = budget * allocation_pct
                allocation_amount = min(allocation_amount, remaining_budget)
                
                if allocation_amount > 0:
                    impressions = int((allocation_amount / format_metric.cpm) * 1000)
                    
                    allocation = FormatAllocation(
                        format_name=format_metric.format_name,
                        budget_allocation=allocation_amount,
                        cpm=format_metric.cpm,
                        estimated_impressions=impressions,
                        recommended_sites=[],
                        notes=f"Balanced allocation (composite score: {score:.3f})"
                    )
                    
                    allocations.append(allocation)
                    remaining_budget -= allocation_amount
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error optimizing balanced allocation: {str(e)}")
            raise
    
    def _apply_allocation_constraints(self, 
                                   allocations: List[FormatAllocation],
                                   constraints: Dict[str, Tuple[float, float]],
                                   total_budget: float) -> List[FormatAllocation]:
        """
        Apply min/max allocation constraints to the allocation list.
        
        Args:
            allocations: List of current allocations
            constraints: Dictionary of format_name -> (min_pct, max_pct) constraints
            total_budget: Total campaign budget
            
        Returns:
            List of constraint-adjusted allocations
        """
        try:
            adjusted_allocations = []
            
            for allocation in allocations:
                if allocation.format_name in constraints:
                    min_pct, max_pct = constraints[allocation.format_name]
                    min_amount = total_budget * min_pct
                    max_amount = total_budget * max_pct
                    
                    # Adjust allocation within constraints
                    adjusted_amount = max(min_amount, min(allocation.budget_allocation, max_amount))
                    
                    if adjusted_amount != allocation.budget_allocation:
                        # Recalculate impressions
                        new_impressions = int((adjusted_amount / allocation.cpm) * 1000)
                        
                        adjusted_allocation = FormatAllocation(
                            format_name=allocation.format_name,
                            budget_allocation=adjusted_amount,
                            cpm=allocation.cpm,
                            estimated_impressions=new_impressions,
                            recommended_sites=allocation.recommended_sites,
                            notes=f"{allocation.notes} (constraint-adjusted)"
                        )
                        
                        adjusted_allocations.append(adjusted_allocation)
                    else:
                        adjusted_allocations.append(allocation)
                else:
                    adjusted_allocations.append(allocation)
            
            return adjusted_allocations
            
        except Exception as e:
            logger.error(f"Error applying allocation constraints: {str(e)}")
            return allocations  # Return original if constraint application fails
    
    def _calculate_allocation_metrics(self, 
                                    allocations: List[FormatAllocation],
                                    format_metrics: List[FormatMetrics],
                                    strategy: OptimizationStrategy) -> AllocationResult:
        """
        Calculate performance metrics for the final allocation.
        
        Args:
            allocations: List of format allocations
            format_metrics: List of format performance metrics
            strategy: Optimization strategy used
            
        Returns:
            AllocationResult with calculated metrics
        """
        try:
            if not allocations:
                raise ValueError("No allocations to calculate metrics for")
            
            # Basic metrics
            total_budget_used = sum(alloc.budget_allocation for alloc in allocations)
            total_impressions = sum(alloc.estimated_impressions for alloc in allocations)
            
            # Estimate total reach (accounting for overlap)
            estimated_reach = self._estimate_total_reach(allocations, format_metrics)
            
            # Calculate average frequency
            average_frequency = total_impressions / estimated_reach if estimated_reach > 0 else 0
            
            # Calculate optimization score based on strategy
            optimization_score = self._calculate_optimization_score(
                allocations, format_metrics, strategy, estimated_reach, average_frequency
            )
            
            # Generate strategy notes
            strategy_notes = self._generate_strategy_notes(allocations, strategy, optimization_score)
            
            return AllocationResult(
                allocations=allocations,
                total_budget_used=total_budget_used,
                estimated_total_reach=estimated_reach,
                estimated_total_impressions=total_impressions,
                average_frequency=average_frequency,
                optimization_score=optimization_score,
                strategy_notes=strategy_notes
            )
            
        except Exception as e:
            logger.error(f"Error calculating allocation metrics: {str(e)}")
            raise
    
    def _estimate_total_reach(self, allocations: List[FormatAllocation], format_metrics: List[FormatMetrics]) -> int:
        """
        Estimate total unique reach accounting for overlap between formats.
        
        Args:
            allocations: List of format allocations
            format_metrics: List of format performance metrics
            
        Returns:
            Estimated total unique reach
        """
        try:
            if not allocations:
                return 0
            
            # Create format lookup
            format_lookup = {fm.format_name: fm for fm in format_metrics}
            
            # Calculate individual format reaches
            format_reaches = []
            for allocation in allocations:
                if allocation.format_name in format_lookup:
                    format_metric = format_lookup[allocation.format_name]
                    impressions_k = allocation.estimated_impressions / 1000
                    format_reach = int(impressions_k * format_metric.estimated_reach_per_1k)
                    format_reaches.append(format_reach)
            
            if not format_reaches:
                return 0
            
            # Simple overlap estimation (can be improved with real data)
            if len(format_reaches) == 1:
                return format_reaches[0]
            
            # Assume 15-25% overlap between formats
            overlap_factor = 0.20  # 20% average overlap
            total_reach_without_overlap = sum(format_reaches)
            estimated_overlap = total_reach_without_overlap * overlap_factor
            
            estimated_total_reach = int(total_reach_without_overlap - estimated_overlap)
            
            return max(estimated_total_reach, max(format_reaches))  # At least as much as largest format
            
        except Exception as e:
            logger.warning(f"Error estimating total reach: {str(e)}")
            return sum(alloc.estimated_impressions for alloc in allocations) // 2  # Fallback estimate
    
    def _calculate_optimization_score(self, 
                                    allocations: List[FormatAllocation],
                                    format_metrics: List[FormatMetrics],
                                    strategy: OptimizationStrategy,
                                    estimated_reach: int,
                                    average_frequency: float) -> float:
        """
        Calculate optimization score based on strategy objectives.
        
        Args:
            allocations: List of format allocations
            format_metrics: List of format performance metrics
            strategy: Optimization strategy used
            estimated_reach: Estimated total reach
            average_frequency: Average frequency
            
        Returns:
            Optimization score (0-100)
        """
        try:
            if not allocations:
                return 0.0
            
            # Create format lookup
            format_lookup = {fm.format_name: fm for fm in format_metrics}
            
            # Calculate strategy-specific scores
            if strategy == OptimizationStrategy.REACH_FOCUSED:
                # Score based on reach efficiency
                total_budget = sum(alloc.budget_allocation for alloc in allocations)
                reach_per_dollar = estimated_reach / total_budget if total_budget > 0 else 0
                score = min(reach_per_dollar * 10, 100)  # Scale to 0-100
                
            elif strategy == OptimizationStrategy.FREQUENCY_FOCUSED:
                # Score based on frequency optimization
                optimal_frequency = self.frequency_sweet_spot
                frequency_score = 100 - abs(average_frequency - optimal_frequency) * 20
                score = max(frequency_score, 0)
                
            elif strategy == OptimizationStrategy.COST_EFFICIENT:
                # Score based on average CPM efficiency
                total_impressions = sum(alloc.estimated_impressions for alloc in allocations)
                total_budget = sum(alloc.budget_allocation for alloc in allocations)
                avg_cpm = (total_budget / total_impressions) * 1000 if total_impressions > 0 else float('inf')
                
                # Lower CPM = higher score
                score = max(100 - (avg_cpm - 1) * 10, 0)  # Assume $1 CPM = 100 points
                
            elif strategy == OptimizationStrategy.HIGH_IMPACT:
                # Score based on weighted impact scores
                total_budget = sum(alloc.budget_allocation for alloc in allocations)
                weighted_impact = 0
                
                for allocation in allocations:
                    if allocation.format_name in format_lookup:
                        format_metric = format_lookup[allocation.format_name]
                        weight = allocation.budget_allocation / total_budget
                        weighted_impact += format_metric.impact_score * weight
                
                score = (weighted_impact / 10) * 100  # Convert to 0-100 scale
                
            else:  # BALANCED
                # Composite score considering multiple factors
                reach_score = min((estimated_reach / 10000) * 25, 25)  # Max 25 points for reach
                frequency_score = max(25 - abs(average_frequency - self.frequency_sweet_spot) * 5, 0)  # Max 25 points
                
                # Cost efficiency component
                total_impressions = sum(alloc.estimated_impressions for alloc in allocations)
                total_budget = sum(alloc.budget_allocation for alloc in allocations)
                avg_cpm = (total_budget / total_impressions) * 1000 if total_impressions > 0 else float('inf')
                cost_score = max(25 - (avg_cpm - 2) * 5, 0)  # Max 25 points
                
                # Impact component
                weighted_impact = 0
                for allocation in allocations:
                    if allocation.format_name in format_lookup:
                        format_metric = format_lookup[allocation.format_name]
                        weight = allocation.budget_allocation / total_budget
                        weighted_impact += format_metric.impact_score * weight
                
                impact_score = (weighted_impact / 10) * 25  # Max 25 points
                
                score = reach_score + frequency_score + cost_score + impact_score
            
            return min(max(score, 0), 100)  # Ensure 0-100 range
            
        except Exception as e:
            logger.warning(f"Error calculating optimization score: {str(e)}")
            return 50.0  # Default neutral score
    
    def _generate_strategy_notes(self, 
                               allocations: List[FormatAllocation],
                               strategy: OptimizationStrategy,
                               optimization_score: float) -> str:
        """
        Generate explanatory notes about the optimization strategy and results.
        
        Args:
            allocations: List of format allocations
            strategy: Optimization strategy used
            optimization_score: Calculated optimization score
            
        Returns:
            Strategy explanation string
        """
        try:
            format_count = len(allocations)
            total_budget = sum(alloc.budget_allocation for alloc in allocations)
            
            strategy_descriptions = {
                OptimizationStrategy.REACH_FOCUSED: 
                    f"Reach-focused strategy prioritizing maximum unique audience coverage. "
                    f"Budget distributed across {format_count} formats to optimize reach efficiency.",
                
                OptimizationStrategy.FREQUENCY_FOCUSED:
                    f"Frequency-focused strategy concentrating budget on fewer formats for deeper engagement. "
                    f"Optimized for {self.frequency_sweet_spot:.1f}x average frequency across {format_count} formats.",
                
                OptimizationStrategy.COST_EFFICIENT:
                    f"Cost-efficient strategy maximizing impressions per dollar. "
                    f"Budget allocated to lowest-CPM formats across {format_count} options.",
                
                OptimizationStrategy.HIGH_IMPACT:
                    f"High-impact strategy prioritizing premium formats and quality placements. "
                    f"Budget focused on highest-scoring formats for maximum campaign impact.",
                
                OptimizationStrategy.BALANCED:
                    f"Balanced strategy optimizing across reach, frequency, cost, and impact metrics. "
                    f"Strategic allocation across {format_count} formats for comprehensive coverage."
            }
            
            base_notes = strategy_descriptions.get(strategy, "Custom optimization strategy applied.")
            
            # Add performance indicator
            if optimization_score >= 80:
                performance_note = " Excellent optimization achieved."
            elif optimization_score >= 60:
                performance_note = " Good optimization achieved."
            elif optimization_score >= 40:
                performance_note = " Moderate optimization achieved."
            else:
                performance_note = " Basic optimization achieved - consider budget increase or strategy adjustment."
            
            return base_notes + performance_note
            
        except Exception as e:
            logger.warning(f"Error generating strategy notes: {str(e)}")
            return f"Applied {strategy.value} optimization strategy across {len(allocations)} formats."