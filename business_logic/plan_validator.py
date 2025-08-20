"""
Plan parsing and validation for AI-generated media plans.

This module handles parsing of AI responses, validation of plan data,
budget verification, and error handling for malformed responses.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from models.data_models import ClientBrief, MediaPlan, FormatAllocation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during plan validation."""
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    plan_index: Optional[int] = None
    allocation_index: Optional[int] = None


@dataclass
class ValidationResult:
    """Result of plan validation process."""
    is_valid: bool
    issues: List[ValidationIssue]
    parsed_plans: List[MediaPlan]
    total_errors: int
    total_warnings: int


class PlanValidator:
    """
    Validates and parses AI-generated media plans.
    
    Handles JSON parsing, data validation, budget verification,
    and conversion to structured MediaPlan objects.
    """
    
    def __init__(self):
        """Initialize the plan validator."""
        # Validation thresholds
        self.budget_tolerance = 0.05  # 5% budget overage tolerance
        self.min_allocation_amount = 100  # Minimum $100 allocation
        self.max_cpm_threshold = 50.0  # Maximum reasonable CPM
        self.min_cpm_threshold = 0.10  # Minimum reasonable CPM
        self.max_impressions_per_dollar = 10000  # Maximum impressions per dollar (sanity check)
        
        # Required fields for validation
        self.required_plan_fields = ['title', 'rationale', 'total_budget', 'allocations']
        self.required_allocation_fields = ['format_name', 'budget_allocation', 'cpm']
        
        # Optional fields with defaults
        self.optional_allocation_fields = {
            'estimated_impressions': 0,
            'recommended_sites': [],
            'notes': ''
        }
    
    def parse_and_validate_plans(self, 
                               ai_response: Union[str, Dict[str, Any]], 
                               client_brief: ClientBrief,
                               available_formats: Dict[str, Any]) -> ValidationResult:
        """
        Parse and validate AI-generated plans from response.
        
        Args:
            ai_response: Raw AI response (JSON string or dict)
            client_brief: Original client brief for validation
            available_formats: Available formats with rate data
            
        Returns:
            ValidationResult with parsed plans and validation issues
        """
        try:
            logger.info("Starting plan parsing and validation")
            
            issues = []
            parsed_plans = []
            
            # Step 1: Parse JSON response
            try:
                plans_data = self._parse_json_response(ai_response)
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Failed to parse JSON response: {str(e)}"
                ))
                return self._create_validation_result(False, issues, [])
            
            # Step 2: Validate response structure
            structure_issues = self._validate_response_structure(plans_data)
            issues.extend(structure_issues)
            
            if any(issue.severity == ValidationSeverity.ERROR for issue in structure_issues):
                return self._create_validation_result(False, issues, [])
            
            # Step 3: Parse and validate individual plans
            raw_plans = plans_data.get('plans', [])
            
            for i, plan_data in enumerate(raw_plans):
                try:
                    plan_issues, media_plan = self._parse_and_validate_single_plan(
                        plan_data, i, client_brief, available_formats
                    )
                    
                    issues.extend(plan_issues)
                    
                    # Only add plan if no critical errors
                    if media_plan and not any(
                        issue.severity == ValidationSeverity.ERROR and issue.plan_index == i 
                        for issue in plan_issues
                    ):
                        parsed_plans.append(media_plan)
                    
                except Exception as e:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Failed to parse plan {i+1}: {str(e)}",
                        plan_index=i
                    ))
            
            # Step 4: Cross-plan validation
            if len(parsed_plans) > 1:
                cross_validation_issues = self._validate_plan_diversity(parsed_plans)
                issues.extend(cross_validation_issues)
            
            # Step 5: Determine overall validation result
            has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
            is_valid = not has_errors and len(parsed_plans) > 0
            
            logger.info(f"Validation complete: {len(parsed_plans)} valid plans, {len(issues)} issues")
            return self._create_validation_result(is_valid, issues, parsed_plans)
            
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Validation process failed: {str(e)}"
            ))
            return self._create_validation_result(False, issues, [])
    
    def _parse_json_response(self, ai_response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse JSON response from AI, handling various formats and errors.
        
        Args:
            ai_response: Raw AI response
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If JSON cannot be parsed
        """
        try:
            if isinstance(ai_response, dict):
                return ai_response
            
            if isinstance(ai_response, str):
                # Clean up common JSON formatting issues
                cleaned_response = self._clean_json_string(ai_response)
                return json.loads(cleaned_response)
            
            raise ValueError(f"Unsupported response type: {type(ai_response)}")
            
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks or other formats
            if isinstance(ai_response, str):
                extracted_json = self._extract_json_from_text(ai_response)
                if extracted_json:
                    return json.loads(extracted_json)
            
            raise ValueError(f"Invalid JSON format: {str(e)}")
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean common JSON formatting issues in AI responses.
        
        Args:
            json_str: Raw JSON string
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code block markers
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        
        # Remove leading/trailing whitespace
        json_str = json_str.strip()
        
        # Fix common quote issues
        json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)  # Add quotes to keys
        
        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON from text that may contain other content.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Extracted JSON string or None
        """
        try:
            # Look for JSON object patterns
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            for match in matches:
                try:
                    # Test if it's valid JSON
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def _validate_response_structure(self, plans_data: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate the overall structure of the AI response.
        
        Args:
            plans_data: Parsed JSON response
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check for 'plans' key
        if 'plans' not in plans_data:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Response missing required 'plans' key",
                field='plans'
            ))
            return issues
        
        plans = plans_data['plans']
        
        # Check plans is a list
        if not isinstance(plans, list):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="'plans' must be a list",
                field='plans'
            ))
            return issues
        
        # Check plans list is not empty
        if len(plans) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No plans found in response",
                field='plans'
            ))
            return issues
        
        # Warn if unexpected number of plans
        if len(plans) != 3:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Expected 3 plans, found {len(plans)}",
                field='plans'
            ))
        
        return issues
    
    def _parse_and_validate_single_plan(self, 
                                      plan_data: Dict[str, Any], 
                                      plan_index: int,
                                      client_brief: ClientBrief,
                                      available_formats: Dict[str, Any]) -> Tuple[List[ValidationIssue], Optional[MediaPlan]]:
        """
        Parse and validate a single media plan.
        
        Args:
            plan_data: Raw plan data dictionary
            plan_index: Index of the plan in the list
            client_brief: Original client brief
            available_formats: Available formats with rate data
            
        Returns:
            Tuple of (validation issues, parsed MediaPlan or None)
        """
        issues = []
        
        try:
            # Validate required fields
            for field in self.required_plan_fields:
                if field not in plan_data:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Missing required field: {field}",
                        field=field,
                        plan_index=plan_index
                    ))
            
            if any(issue.severity == ValidationSeverity.ERROR for issue in issues):
                return issues, None
            
            # Validate and parse basic fields
            title = str(plan_data['title']).strip()
            if not title:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Plan title cannot be empty",
                    field='title',
                    plan_index=plan_index
                ))
            
            rationale = str(plan_data['rationale']).strip()
            if not rationale:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Plan rationale is empty",
                    field='rationale',
                    plan_index=plan_index
                ))
            
            # Validate budget
            try:
                total_budget = float(plan_data['total_budget'])
                if total_budget <= 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Plan budget must be positive",
                        field='total_budget',
                        plan_index=plan_index
                    ))
                elif total_budget > client_brief.budget * (1 + self.budget_tolerance):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Plan budget ${total_budget:,.2f} exceeds client budget ${client_brief.budget:,.2f}",
                        field='total_budget',
                        plan_index=plan_index
                    ))
            except (ValueError, TypeError):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Invalid budget format",
                    field='total_budget',
                    plan_index=plan_index
                ))
                return issues, None
            
            # Validate allocations
            allocations_data = plan_data.get('allocations', [])
            if not isinstance(allocations_data, list) or len(allocations_data) == 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Plan must have at least one allocation",
                    field='allocations',
                    plan_index=plan_index
                ))
                return issues, None
            
            # Parse and validate individual allocations
            allocation_issues, parsed_allocations = self._parse_allocations(
                allocations_data, plan_index, client_brief, available_formats
            )
            issues.extend(allocation_issues)
            
            if not parsed_allocations:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="No valid allocations found in plan",
                    field='allocations',
                    plan_index=plan_index
                ))
                return issues, None
            
            # Validate total allocation vs plan budget
            total_allocated = sum(alloc.budget_allocation for alloc in parsed_allocations)
            budget_diff = abs(total_allocated - total_budget)
            
            if budget_diff > total_budget * 0.01:  # 1% tolerance
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Allocation total ${total_allocated:,.2f} doesn't match plan budget ${total_budget:,.2f}",
                    field='total_budget',
                    plan_index=plan_index
                ))
                # Use actual allocated amount
                total_budget = total_allocated
            
            # Calculate estimated metrics
            estimated_reach = int(plan_data.get('estimated_reach', 0))
            estimated_impressions = sum(alloc.estimated_impressions for alloc in parsed_allocations)
            
            # Validate estimated impressions consistency
            if 'estimated_impressions' in plan_data:
                plan_impressions = int(plan_data['estimated_impressions'])
                if abs(plan_impressions - estimated_impressions) > estimated_impressions * 0.1:  # 10% tolerance
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Plan impressions {plan_impressions:,} don't match allocation total {estimated_impressions:,}",
                        field='estimated_impressions',
                        plan_index=plan_index
                    ))
            
            # Create MediaPlan object
            media_plan = MediaPlan(
                plan_id=f"plan_{plan_index + 1}_{int(datetime.now().timestamp())}",
                title=title,
                total_budget=total_budget,
                allocations=parsed_allocations,
                estimated_reach=estimated_reach,
                estimated_impressions=estimated_impressions,
                rationale=rationale,
                created_at=datetime.now()
            )
            
            return issues, media_plan
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Error parsing plan: {str(e)}",
                plan_index=plan_index
            ))
            return issues, None
    
    def _parse_allocations(self, 
                         allocations_data: List[Dict[str, Any]], 
                         plan_index: int,
                         client_brief: ClientBrief,
                         available_formats: Dict[str, Any]) -> Tuple[List[ValidationIssue], List[FormatAllocation]]:
        """
        Parse and validate format allocations.
        
        Args:
            allocations_data: List of allocation dictionaries
            plan_index: Index of the parent plan
            client_brief: Original client brief
            available_formats: Available formats with rate data
            
        Returns:
            Tuple of (validation issues, parsed allocations)
        """
        issues = []
        parsed_allocations = []
        
        for alloc_index, alloc_data in enumerate(allocations_data):
            try:
                # Validate required fields
                for field in self.required_allocation_fields:
                    if field not in alloc_data:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Allocation missing required field: {field}",
                            field=field,
                            plan_index=plan_index,
                            allocation_index=alloc_index
                        ))
                        continue
                
                # Parse format name
                format_name = str(alloc_data['format_name']).strip()
                if not format_name:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Format name cannot be empty",
                        field='format_name',
                        plan_index=plan_index,
                        allocation_index=alloc_index
                    ))
                    continue
                
                # Validate format exists in available formats
                if not self._is_format_available(format_name, available_formats):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Format '{format_name}' not found in available formats",
                        field='format_name',
                        plan_index=plan_index,
                        allocation_index=alloc_index
                    ))
                
                # Validate format is in client selection (if specified)
                if client_brief.selected_formats and format_name not in client_brief.selected_formats:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Format '{format_name}' not in client's selected formats",
                        field='format_name',
                        plan_index=plan_index,
                        allocation_index=alloc_index
                    ))
                    continue
                
                # Parse and validate budget allocation
                try:
                    budget_allocation = float(alloc_data['budget_allocation'])
                    if budget_allocation < self.min_allocation_amount:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Allocation ${budget_allocation:,.2f} below minimum ${self.min_allocation_amount}",
                            field='budget_allocation',
                            plan_index=plan_index,
                            allocation_index=alloc_index
                        ))
                except (ValueError, TypeError):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Invalid budget allocation format",
                        field='budget_allocation',
                        plan_index=plan_index,
                        allocation_index=alloc_index
                    ))
                    continue
                
                # Parse and validate CPM
                try:
                    cpm = float(alloc_data['cpm'])
                    if cpm < self.min_cpm_threshold:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"CPM ${cpm:.2f} unusually low (minimum ${self.min_cpm_threshold:.2f})",
                            field='cpm',
                            plan_index=plan_index,
                            allocation_index=alloc_index
                        ))
                    elif cpm > self.max_cpm_threshold:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"CPM ${cpm:.2f} unusually high (maximum ${self.max_cpm_threshold:.2f})",
                            field='cpm',
                            plan_index=plan_index,
                            allocation_index=alloc_index
                        ))
                except (ValueError, TypeError):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Invalid CPM format",
                        field='cpm',
                        plan_index=plan_index,
                        allocation_index=alloc_index
                    ))
                    continue
                
                # Parse optional fields with defaults
                estimated_impressions = int(alloc_data.get('estimated_impressions', 0))
                
                # Calculate impressions if not provided or validate if provided
                calculated_impressions = int((budget_allocation / cpm) * 1000) if cpm > 0 else 0
                
                if estimated_impressions == 0:
                    estimated_impressions = calculated_impressions
                elif abs(estimated_impressions - calculated_impressions) > calculated_impressions * 0.1:  # 10% tolerance
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Estimated impressions {estimated_impressions:,} don't match calculation {calculated_impressions:,}",
                        field='estimated_impressions',
                        plan_index=plan_index,
                        allocation_index=alloc_index
                    ))
                    estimated_impressions = calculated_impressions  # Use calculated value
                
                # Validate impressions per dollar ratio
                impressions_per_dollar = estimated_impressions / budget_allocation if budget_allocation > 0 else 0
                if impressions_per_dollar > self.max_impressions_per_dollar:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Impressions per dollar ({impressions_per_dollar:.0f}) seems unrealistic",
                        field='estimated_impressions',
                        plan_index=plan_index,
                        allocation_index=alloc_index
                    ))
                
                # Parse sites and notes
                recommended_sites = alloc_data.get('recommended_sites', [])
                if not isinstance(recommended_sites, list):
                    recommended_sites = []
                
                notes = str(alloc_data.get('notes', '')).strip()
                
                # Create FormatAllocation object
                allocation = FormatAllocation(
                    format_name=format_name,
                    budget_allocation=budget_allocation,
                    cpm=cpm,
                    estimated_impressions=estimated_impressions,
                    recommended_sites=recommended_sites,
                    notes=notes
                )
                
                parsed_allocations.append(allocation)
                
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Error parsing allocation {alloc_index + 1}: {str(e)}",
                    plan_index=plan_index,
                    allocation_index=alloc_index
                ))
        
        return issues, parsed_allocations
    
    def _is_format_available(self, format_name: str, available_formats: Dict[str, Any]) -> bool:
        """
        Check if a format is available in the market data.
        
        Args:
            format_name: Name of the format to check
            available_formats: Available formats dictionary
            
        Returns:
            True if format is available, False otherwise
        """
        try:
            # Check in impact formats
            impact_formats = available_formats.get('impact_formats', {})
            if format_name in impact_formats:
                return True
            
            # Check in reach formats
            reach_formats = available_formats.get('reach_formats', {})
            if format_name in reach_formats:
                return True
            
            # Check in combined formats list
            all_formats = available_formats.get('formats', [])
            if format_name in all_formats:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _validate_plan_diversity(self, plans: List[MediaPlan]) -> List[ValidationIssue]:
        """
        Validate that plans have sufficient strategic diversity.
        
        Args:
            plans: List of parsed MediaPlan objects
            
        Returns:
            List of validation issues related to plan diversity
        """
        issues = []
        
        try:
            if len(plans) < 2:
                return issues
            
            # Check for duplicate titles
            titles = [plan.title for plan in plans]
            if len(set(titles)) != len(titles):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Some plans have duplicate or very similar titles"
                ))
            
            # Check for similar budget allocations
            for i in range(len(plans)):
                for j in range(i + 1, len(plans)):
                    similarity = self._calculate_allocation_similarity(plans[i], plans[j])
                    
                    if similarity > 0.85:  # 85% similarity threshold
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Plans '{plans[i].title}' and '{plans[j].title}' are very similar ({similarity:.1%})"
                        ))
            
            return issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Error validating plan diversity: {str(e)}"
            ))
            return issues
    
    def _calculate_allocation_similarity(self, plan1: MediaPlan, plan2: MediaPlan) -> float:
        """
        Calculate similarity between two plans based on their allocations.
        
        Args:
            plan1: First MediaPlan
            plan2: Second MediaPlan
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Create allocation dictionaries
            alloc1 = {alloc.format_name: alloc.budget_allocation for alloc in plan1.allocations}
            alloc2 = {alloc.format_name: alloc.budget_allocation for alloc in plan2.allocations}
            
            # Get all formats
            all_formats = set(alloc1.keys()) | set(alloc2.keys())
            
            if not all_formats:
                return 1.0  # Both empty
            
            # Calculate similarity
            total_similarity = 0.0
            max_budget = max(plan1.total_budget, plan2.total_budget)
            
            for format_name in all_formats:
                budget1 = alloc1.get(format_name, 0) / max_budget
                budget2 = alloc2.get(format_name, 0) / max_budget
                format_similarity = 1.0 - abs(budget1 - budget2)
                total_similarity += format_similarity
            
            return total_similarity / len(all_formats)
            
        except Exception:
            return 0.0  # Assume different if calculation fails
    
    def _create_validation_result(self, 
                                is_valid: bool, 
                                issues: List[ValidationIssue], 
                                parsed_plans: List[MediaPlan]) -> ValidationResult:
        """
        Create a ValidationResult object with summary statistics.
        
        Args:
            is_valid: Whether validation passed overall
            issues: List of validation issues
            parsed_plans: List of successfully parsed plans
            
        Returns:
            ValidationResult object
        """
        total_errors = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        total_warnings = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            parsed_plans=parsed_plans,
            total_errors=total_errors,
            total_warnings=total_warnings
        )
    
    def validate_budget_constraints(self, 
                                  plans: List[MediaPlan], 
                                  client_brief: ClientBrief) -> List[ValidationIssue]:
        """
        Validate that all plans meet budget constraints.
        
        Args:
            plans: List of MediaPlan objects to validate
            client_brief: Original client brief with budget constraints
            
        Returns:
            List of budget-related validation issues
        """
        issues = []
        
        try:
            for i, plan in enumerate(plans):
                # Check total budget constraint
                if plan.total_budget > client_brief.budget * (1 + self.budget_tolerance):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Plan exceeds budget: ${plan.total_budget:,.2f} > ${client_brief.budget:,.2f}",
                        field='total_budget',
                        plan_index=i
                    ))
                
                # Check minimum budget utilization
                if plan.total_budget < client_brief.budget * 0.5:  # Less than 50% utilization
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Plan uses only {(plan.total_budget/client_brief.budget)*100:.1f}% of available budget",
                        field='total_budget',
                        plan_index=i
                    ))
                
                # Validate allocation totals match plan budget
                total_allocated = sum(alloc.budget_allocation for alloc in plan.allocations)
                if abs(total_allocated - plan.total_budget) > plan.total_budget * 0.01:  # 1% tolerance
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Allocation total ${total_allocated:,.2f} doesn't match plan budget ${plan.total_budget:,.2f}",
                        field='allocations',
                        plan_index=i
                    ))
            
            return issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Error validating budget constraints: {str(e)}"
            ))
            return issues
    
    def get_validation_summary(self, validation_result: ValidationResult) -> str:
        """
        Generate a human-readable summary of validation results.
        
        Args:
            validation_result: ValidationResult object
            
        Returns:
            Formatted summary string
        """
        try:
            summary_lines = []
            
            # Overall status
            status = "PASSED" if validation_result.is_valid else "FAILED"
            summary_lines.append(f"Validation Status: {status}")
            
            # Plan count
            summary_lines.append(f"Valid Plans: {len(validation_result.parsed_plans)}")
            
            # Issue summary
            if validation_result.total_errors > 0:
                summary_lines.append(f"Errors: {validation_result.total_errors}")
            
            if validation_result.total_warnings > 0:
                summary_lines.append(f"Warnings: {validation_result.total_warnings}")
            
            # Detailed issues
            if validation_result.issues:
                summary_lines.append("\nIssues:")
                for issue in validation_result.issues:
                    prefix = "ERROR" if issue.severity == ValidationSeverity.ERROR else "WARNING"
                    location = ""
                    
                    if issue.plan_index is not None:
                        location += f" [Plan {issue.plan_index + 1}"
                        if issue.allocation_index is not None:
                            location += f", Allocation {issue.allocation_index + 1}"
                        location += "]"
                    
                    summary_lines.append(f"  {prefix}{location}: {issue.message}")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            return f"Error generating validation summary: {str(e)}"