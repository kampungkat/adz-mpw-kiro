"""
Core data models for the AI Media Planner application.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class ClientBrief:
    """Client brief containing campaign requirements and preferences."""
    brand_name: str
    budget: float
    country: str
    campaign_period: str
    objective: str
    planning_mode: str
    selected_formats: Optional[List[str]] = None


@dataclass
class FormatAllocation:
    """Budget allocation for a specific ad format."""
    format_name: str
    budget_allocation: float
    cpm: float
    estimated_impressions: int
    recommended_sites: List[str]
    notes: str


@dataclass
class MediaPlan:
    """Complete media plan with allocations and estimates."""
    plan_id: str
    title: str
    total_budget: float
    allocations: List[FormatAllocation]
    estimated_reach: int
    estimated_impressions: int
    rationale: str
    created_at: datetime


@dataclass
class RateCard:
    """Rate card data for a specific market."""
    market: str
    format_rates: Dict[str, float]
    last_updated: datetime
    reach_tiers: Dict[str, float]


@dataclass
class SiteData:
    """Site categorization data for a specific market."""
    market: str
    sites_by_format: Dict[str, List[str]]
    categories: Dict[str, List[str]]
    last_updated: datetime


@dataclass
class TrainingData:
    """Training data for model fine-tuning."""
    campaign_brief: str
    generated_plan: str
    performance_metrics: Optional[Dict[str, float]]
    created_at: datetime
    validated: bool


@dataclass
class FineTuningJob:
    """Fine-tuning job tracking."""
    job_id: str
    model_name: str
    training_file_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    fine_tuned_model: Optional[str]


@dataclass
class AudienceSegment:
    """Audience segment for targeting (future enhancement)."""
    segment_id: str
    name: str
    demographics: Dict[str, Any]
    interests: List[str]
    market: str
    estimated_size: int
    recommended_formats: List[str]


@dataclass
class TargetingData:
    """Targeting data with audience segments (future enhancement)."""
    primary_segment: AudienceSegment
    secondary_segments: List[AudienceSegment]
    overlap_analysis: Dict[str, float]