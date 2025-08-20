"""
Model Training Manager for collecting training data and managing OpenAI fine-tuning.

This module handles the collection of historical campaign briefs and successful plans,
formats data for OpenAI fine-tuning, and manages the fine-tuning workflow.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import openai
from openai import OpenAI

from models.data_models import ClientBrief, MediaPlan, TrainingData, FineTuningJob
from config.settings import config_manager
from .error_handler import error_handler, RetryConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainingManager:
    """
    Manages model training data collection and OpenAI fine-tuning workflows.
    
    Handles collection of historical campaign data, validation and formatting
    for OpenAI fine-tuning, and management of training jobs.
    """
    
    def __init__(self, skip_openai_init: bool = False):
        """
        Initialize the Model Training Manager.
        
        Args:
            skip_openai_init: Skip OpenAI client initialization (for testing)
        """
        self.client = None
        if not skip_openai_init:
            self._initialize_openai_client()
        
        # Training data storage
        self.training_data_dir = "training_data"
        self.training_data_file = os.path.join(self.training_data_dir, "collected_data.json")
        self.openai_format_file = os.path.join(self.training_data_dir, "openai_training_data.jsonl")
        
        # Fine-tuning configuration
        self.min_training_examples = 10  # OpenAI minimum
        self.max_training_examples = 1000  # Practical limit
        self.validation_split = 0.1  # 10% for validation
        
        # Create training data directory if it doesn't exist
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        # Initialize training data storage
        self.training_data: List[TrainingData] = []
        self._load_existing_training_data()
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client with API key."""
        try:
            api_key = config_manager.get_openai_api_key()
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized for training manager")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    def _load_existing_training_data(self):
        """Load existing training data from storage."""
        try:
            if os.path.exists(self.training_data_file):
                with open(self.training_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.training_data = []
                for item in data:
                    training_data = TrainingData(
                        campaign_brief=item['campaign_brief'],
                        generated_plan=item['generated_plan'],
                        performance_metrics=item.get('performance_metrics'),
                        created_at=datetime.fromisoformat(item['created_at']),
                        validated=item.get('validated', False)
                    )
                    self.training_data.append(training_data)
                
                logger.info(f"Loaded {len(self.training_data)} existing training examples")
            else:
                logger.info("No existing training data found")
                
        except Exception as e:
            logger.error(f"Error loading existing training data: {str(e)}")
            self.training_data = []
    
    def collect_training_data(self, client_brief: ClientBrief, generated_plan: MediaPlan, 
                            performance_metrics: Optional[Dict[str, float]] = None,
                            validated: bool = False) -> bool:
        """
        Collect and store a campaign brief and successful plan for training.
        
        Args:
            client_brief: The original client brief
            generated_plan: The successful media plan
            performance_metrics: Optional performance metrics for the plan
            validated: Whether this plan has been validated as successful
            
        Returns:
            True if data was successfully collected, False otherwise
        """
        try:
            # Format campaign brief for training
            brief_text = self._format_campaign_brief(client_brief)
            
            # Format generated plan for training
            plan_text = self._format_generated_plan(generated_plan)
            
            # Create training data entry
            training_entry = TrainingData(
                campaign_brief=brief_text,
                generated_plan=plan_text,
                performance_metrics=performance_metrics,
                created_at=datetime.now(),
                validated=validated
            )
            
            # Validate training data quality
            if not self._validate_training_data_quality(training_entry):
                logger.warning("Training data failed quality validation")
                return False
            
            # Add to collection
            self.training_data.append(training_entry)
            
            # Save to storage
            self._save_training_data()
            
            logger.info(f"Collected training data: {client_brief.brand_name} - {generated_plan.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting training data: {str(e)}")
            return False
    
    def _format_campaign_brief(self, client_brief: ClientBrief) -> str:
        """
        Format a ClientBrief into a standardized text format for training.
        
        Args:
            client_brief: The client brief to format
            
        Returns:
            Formatted brief text
        """
        try:
            brief_parts = [
                f"Brand: {client_brief.brand_name}",
                f"Budget: ${client_brief.budget:,.2f}",
                f"Market: {client_brief.country}",
                f"Campaign Period: {client_brief.campaign_period}",
                f"Objective: {client_brief.objective}",
                f"Planning Mode: {client_brief.planning_mode}"
            ]
            
            if client_brief.selected_formats:
                brief_parts.append(f"Selected Formats: {', '.join(client_brief.selected_formats)}")
            
            return "\n".join(brief_parts)
            
        except Exception as e:
            logger.error(f"Error formatting campaign brief: {str(e)}")
            return ""
    
    def _format_generated_plan(self, media_plan: MediaPlan) -> str:
        """
        Format a MediaPlan into a standardized text format for training.
        
        Args:
            media_plan: The media plan to format
            
        Returns:
            Formatted plan text
        """
        try:
            plan_parts = [
                f"Title: {media_plan.title}",
                f"Rationale: {media_plan.rationale}",
                f"Total Budget: ${media_plan.total_budget:,.2f}",
                f"Estimated Reach: {media_plan.estimated_reach:,}",
                f"Estimated Impressions: {media_plan.estimated_impressions:,}",
                "",
                "Allocations:"
            ]
            
            for allocation in media_plan.allocations:
                alloc_text = [
                    f"  - Format: {allocation.format_name}",
                    f"    Budget: ${allocation.budget_allocation:,.2f}",
                    f"    CPM: ${allocation.cpm:.2f}",
                    f"    Impressions: {allocation.estimated_impressions:,}",
                    f"    Sites: {', '.join(allocation.recommended_sites[:3])}{'...' if len(allocation.recommended_sites) > 3 else ''}",
                    f"    Notes: {allocation.notes}"
                ]
                plan_parts.extend(alloc_text)
            
            return "\n".join(plan_parts)
            
        except Exception as e:
            logger.error(f"Error formatting generated plan: {str(e)}")
            return ""
    
    def _validate_training_data_quality(self, training_data: TrainingData) -> bool:
        """
        Validate the quality of training data before adding to collection.
        
        Args:
            training_data: The training data to validate
            
        Returns:
            True if data meets quality requirements, False otherwise
        """
        try:
            # Check minimum content requirements
            if not training_data.campaign_brief or len(training_data.campaign_brief.strip()) < 50:
                logger.warning("Campaign brief too short or empty")
                return False
            
            if not training_data.generated_plan or len(training_data.generated_plan.strip()) < 100:
                logger.warning("Generated plan too short or empty")
                return False
            
            # Check for required fields in brief
            required_brief_fields = ["Brand:", "Budget:", "Market:", "Objective:"]
            for field in required_brief_fields:
                if field not in training_data.campaign_brief:
                    logger.warning(f"Missing required field in brief: {field}")
                    return False
            
            # Check for required fields in plan
            required_plan_fields = ["Title:", "Rationale:", "Total Budget:", "Allocations:"]
            for field in required_plan_fields:
                if field not in training_data.generated_plan:
                    logger.warning(f"Missing required field in plan: {field}")
                    return False
            
            # Check for duplicate data
            for existing_data in self.training_data:
                if (existing_data.campaign_brief == training_data.campaign_brief and 
                    existing_data.generated_plan == training_data.generated_plan):
                    logger.warning("Duplicate training data detected")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating training data quality: {str(e)}")
            return False
    
    def _save_training_data(self):
        """Save training data to persistent storage."""
        try:
            data_to_save = []
            for training_data in self.training_data:
                data_to_save.append({
                    'campaign_brief': training_data.campaign_brief,
                    'generated_plan': training_data.generated_plan,
                    'performance_metrics': training_data.performance_metrics,
                    'created_at': training_data.created_at.isoformat(),
                    'validated': training_data.validated
                })
            
            with open(self.training_data_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(data_to_save)} training examples to storage")
            
        except Exception as e:
            logger.error(f"Error saving training data: {str(e)}")
            raise
    
    def export_training_data_for_openai(self, include_unvalidated: bool = False,
                                      max_examples: Optional[int] = None) -> Tuple[bool, str]:
        """
        Export training data in OpenAI fine-tuning format (JSONL).
        
        Args:
            include_unvalidated: Whether to include unvalidated training examples
            max_examples: Maximum number of examples to export (None for all)
            
        Returns:
            Tuple of (success, file_path or error_message)
        """
        try:
            # Filter training data
            filtered_data = []
            for training_data in self.training_data:
                if not include_unvalidated and not training_data.validated:
                    continue
                filtered_data.append(training_data)
            
            if len(filtered_data) < self.min_training_examples:
                return False, f"Insufficient training examples: {len(filtered_data)} < {self.min_training_examples}"
            
            # Limit examples if specified
            if max_examples and len(filtered_data) > max_examples:
                # Sort by validation status and creation date (validated and recent first)
                filtered_data.sort(key=lambda x: (x.validated, x.created_at), reverse=True)
                filtered_data = filtered_data[:max_examples]
            
            # Convert to OpenAI format
            openai_examples = []
            for training_data in filtered_data:
                example = self._convert_to_openai_format(training_data)
                if example and self._validate_openai_format(example):
                    openai_examples.append(example)
            
            if len(openai_examples) < self.min_training_examples:
                return False, f"Insufficient valid examples after conversion: {len(openai_examples)}"
            
            # Write to JSONL file
            with open(self.openai_format_file, 'w', encoding='utf-8') as f:
                for example in openai_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            logger.info(f"Exported {len(openai_examples)} training examples to {self.openai_format_file}")
            return True, self.openai_format_file
            
        except Exception as e:
            error_msg = f"Error exporting training data: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _convert_to_openai_format(self, training_data: TrainingData) -> Optional[Dict[str, Any]]:
        """
        Convert training data to OpenAI fine-tuning format.
        
        Args:
            training_data: The training data to convert
            
        Returns:
            OpenAI format dictionary or None if conversion fails
        """
        try:
            # Create system message for media planning context
            system_message = """You are an expert media planner at Adzymic, a leading digital advertising agency. Your task is to create strategic media plans that optimize reach, frequency, and budget efficiency for clients. Generate detailed media plans with specific budget allocations, format selections, and strategic rationale."""
            
            # Create user message from campaign brief
            user_message = f"Create a media plan for this campaign:\n\n{training_data.campaign_brief}"
            
            # Create assistant message from generated plan
            assistant_message = training_data.generated_plan
            
            # Format for OpenAI fine-tuning
            openai_example = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ]
            }
            
            return openai_example
            
        except Exception as e:
            logger.error(f"Error converting to OpenAI format: {str(e)}")
            return None
    
    def _validate_openai_format(self, example: Dict[str, Any]) -> bool:
        """
        Validate that an example meets OpenAI fine-tuning requirements.
        
        Args:
            example: The example to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required structure
            if 'messages' not in example:
                return False
            
            messages = example['messages']
            if not isinstance(messages, list) or len(messages) != 3:
                return False
            
            # Check message roles
            expected_roles = ['system', 'user', 'assistant']
            for i, message in enumerate(messages):
                if not isinstance(message, dict):
                    return False
                if message.get('role') != expected_roles[i]:
                    return False
                if not message.get('content') or len(message['content'].strip()) < 10:
                    return False
            
            # Check content length limits (OpenAI has token limits)
            total_content_length = sum(len(msg['content']) for msg in messages)
            if total_content_length > 16000:  # Conservative limit for token count
                logger.warning("Example content too long, may exceed token limits")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating OpenAI format: {str(e)}")
            return False
    
    def validate_training_data_requirements(self) -> Dict[str, Any]:
        """
        Validate that collected training data meets OpenAI requirements.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        try:
            total_examples = len(self.training_data)
            validated_examples = sum(1 for data in self.training_data if data.validated)
            
            # Check minimum requirements
            meets_minimum = total_examples >= self.min_training_examples
            has_validated = validated_examples > 0
            
            # Calculate data quality metrics
            avg_brief_length = sum(len(data.campaign_brief) for data in self.training_data) / max(total_examples, 1)
            avg_plan_length = sum(len(data.generated_plan) for data in self.training_data) / max(total_examples, 1)
            
            # Check data diversity (simple check for unique brands)
            unique_brands = set()
            for data in self.training_data:
                if "Brand:" in data.campaign_brief:
                    brand_line = [line for line in data.campaign_brief.split('\n') if line.startswith('Brand:')]
                    if brand_line:
                        unique_brands.add(brand_line[0])
            
            diversity_score = len(unique_brands) / max(total_examples, 1)
            
            # Generate recommendations
            recommendations = []
            if not meets_minimum:
                recommendations.append(f"Need at least {self.min_training_examples - total_examples} more training examples")
            
            if validated_examples < total_examples * 0.5:
                recommendations.append("Consider validating more training examples for better quality")
            
            if diversity_score < 0.3:
                recommendations.append("Training data lacks diversity - collect examples from different brands/campaigns")
            
            if avg_brief_length < 100:
                recommendations.append("Campaign briefs are too short - include more detailed requirements")
            
            if avg_plan_length < 200:
                recommendations.append("Generated plans are too short - include more detailed allocations and rationale")
            
            return {
                'total_examples': total_examples,
                'validated_examples': validated_examples,
                'meets_minimum_requirements': meets_minimum,
                'has_validated_examples': has_validated,
                'average_brief_length': avg_brief_length,
                'average_plan_length': avg_plan_length,
                'diversity_score': diversity_score,
                'recommendations': recommendations,
                'ready_for_fine_tuning': meets_minimum and has_validated and diversity_score > 0.2
            }
            
        except Exception as e:
            logger.error(f"Error validating training data requirements: {str(e)}")
            return {
                'error': str(e),
                'ready_for_fine_tuning': False
            }
    
    def get_training_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of collected training data.
        
        Returns:
            Dictionary with training data statistics
        """
        try:
            if not self.training_data:
                return {
                    'total_examples': 0,
                    'validated_examples': 0,
                    'date_range': None,
                    'ready_for_export': False
                }
            
            total_examples = len(self.training_data)
            validated_examples = sum(1 for data in self.training_data if data.validated)
            
            # Date range
            dates = [data.created_at for data in self.training_data]
            min_date = min(dates)
            max_date = max(dates)
            
            # Recent activity (last 30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_examples = sum(1 for data in self.training_data if data.created_at > thirty_days_ago)
            
            return {
                'total_examples': total_examples,
                'validated_examples': validated_examples,
                'unvalidated_examples': total_examples - validated_examples,
                'date_range': {
                    'earliest': min_date.isoformat(),
                    'latest': max_date.isoformat()
                },
                'recent_examples_30_days': recent_examples,
                'ready_for_export': total_examples >= self.min_training_examples,
                'storage_file': self.training_data_file,
                'openai_export_file': self.openai_format_file
            }
            
        except Exception as e:
            logger.error(f"Error getting training data summary: {str(e)}")
            return {'error': str(e)}
    
    def mark_training_data_validated(self, brief_text: str, plan_text: str, 
                                   performance_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Mark specific training data as validated with optional performance metrics.
        
        Args:
            brief_text: The campaign brief text to find
            plan_text: The generated plan text to find
            performance_metrics: Optional performance metrics to add
            
        Returns:
            True if data was found and marked as validated, False otherwise
        """
        try:
            for training_data in self.training_data:
                if (training_data.campaign_brief == brief_text and 
                    training_data.generated_plan == plan_text):
                    
                    training_data.validated = True
                    if performance_metrics:
                        training_data.performance_metrics = performance_metrics
                    
                    # Save updated data
                    self._save_training_data()
                    
                    logger.info("Marked training data as validated")
                    return True
            
            logger.warning("Training data not found for validation")
            return False
            
        except Exception as e:
            logger.error(f"Error marking training data as validated: {str(e)}")
            return False
    
    def clear_training_data(self, confirmed: bool = False) -> bool:
        """
        Clear all collected training data (use with caution).
        
        Args:
            confirmed: Must be True to actually clear data
            
        Returns:
            True if data was cleared, False otherwise
        """
        try:
            if not confirmed:
                logger.warning("Clear training data called without confirmation")
                return False
            
            self.training_data = []
            self._save_training_data()
            
            # Remove OpenAI export file if it exists
            if os.path.exists(self.openai_format_file):
                os.remove(self.openai_format_file)
            
            logger.info("All training data cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing training data: {str(e)}")
            return False
    
    # Fine-tuning job management methods
    
    def initiate_fine_tuning_job(self, model_name: str = "gpt-3.5-turbo", 
                               training_file_path: Optional[str] = None,
                               validation_file_path: Optional[str] = None,
                               hyperparameters: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Initiate a fine-tuning job with OpenAI.
        
        Args:
            model_name: Base model to fine-tune (default: gpt-3.5-turbo)
            training_file_path: Path to training data file (uses exported data if None)
            validation_file_path: Path to validation data file (optional)
            hyperparameters: Custom hyperparameters for fine-tuning
            
        Returns:
            Tuple of (success, job_id or error_message)
        """
        try:
            if not self.client:
                return False, "OpenAI client not initialized"
            
            # Use exported training data if no file path provided
            if not training_file_path:
                success, file_path = self.export_training_data_for_openai(include_unvalidated=False)
                if not success:
                    return False, f"Failed to export training data: {file_path}"
                training_file_path = file_path
            
            # Upload training file
            logger.info(f"Uploading training file: {training_file_path}")
            with open(training_file_path, 'rb') as f:
                training_file = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            logger.info(f"Training file uploaded with ID: {training_file.id}")
            
            # Upload validation file if provided
            validation_file_id = None
            if validation_file_path and os.path.exists(validation_file_path):
                logger.info(f"Uploading validation file: {validation_file_path}")
                with open(validation_file_path, 'rb') as f:
                    validation_file = self.client.files.create(
                        file=f,
                        purpose='fine-tune'
                    )
                validation_file_id = validation_file.id
                logger.info(f"Validation file uploaded with ID: {validation_file_id}")
            
            # Set up fine-tuning parameters
            fine_tune_params = {
                'training_file': training_file.id,
                'model': model_name
            }
            
            if validation_file_id:
                fine_tune_params['validation_file'] = validation_file_id
            
            # Add custom hyperparameters if provided
            if hyperparameters:
                fine_tune_params['hyperparameters'] = hyperparameters
            
            # Create fine-tuning job
            logger.info("Creating fine-tuning job...")
            fine_tune_job = self.client.fine_tuning.jobs.create(**fine_tune_params)
            
            # Store job information
            job_info = FineTuningJob(
                job_id=fine_tune_job.id,
                model_name=model_name,
                training_file_id=training_file.id,
                status=fine_tune_job.status,
                created_at=datetime.now(),
                completed_at=None,
                fine_tuned_model=None
            )
            
            self._save_fine_tuning_job(job_info)
            
            logger.info(f"Fine-tuning job created with ID: {fine_tune_job.id}")
            return True, fine_tune_job.id
            
        except openai.AuthenticationError as e:
            error_msg = f"Authentication failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
            
        except openai.RateLimitError as e:
            error_msg = f"Rate limit exceeded: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
            
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Error initiating fine-tuning job: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def monitor_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        """
        Monitor the status and progress of a fine-tuning job.
        
        Args:
            job_id: The fine-tuning job ID
            
        Returns:
            Dictionary with job status and progress information
        """
        try:
            if not self.client:
                return {'error': 'OpenAI client not initialized'}
            
            # Get job status from OpenAI
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            
            # Update stored job information
            self._update_fine_tuning_job_status(job_id, job.status, job.fine_tuned_model)
            
            # Get training events for progress tracking
            events = []
            try:
                events_response = self.client.fine_tuning.jobs.list_events(job_id, limit=10)
                events = [
                    {
                        'timestamp': event.created_at,
                        'level': event.level,
                        'message': event.message
                    }
                    for event in events_response.data
                ]
            except Exception as e:
                logger.warning(f"Could not retrieve job events: {str(e)}")
            
            # Calculate progress estimate
            progress_estimate = self._estimate_job_progress(job.status, events)
            
            return {
                'job_id': job_id,
                'status': job.status,
                'model': job.model,
                'created_at': job.created_at,
                'finished_at': job.finished_at,
                'fine_tuned_model': job.fine_tuned_model,
                'training_file': job.training_file,
                'validation_file': job.validation_file,
                'hyperparameters': job.hyperparameters,
                'result_files': job.result_files,
                'trained_tokens': job.trained_tokens,
                'progress_estimate': progress_estimate,
                'recent_events': events,
                'error': job.error
            }
            
        except openai.NotFoundError:
            return {'error': f'Fine-tuning job {job_id} not found'}
            
        except Exception as e:
            error_msg = f"Error monitoring fine-tuning job: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg}
    
    def _estimate_job_progress(self, status: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate the progress of a fine-tuning job based on status and events.
        
        Args:
            status: Current job status
            events: Recent job events
            
        Returns:
            Dictionary with progress information
        """
        try:
            progress_map = {
                'validating_files': 10,
                'queued': 20,
                'running': 50,
                'succeeded': 100,
                'failed': 0,
                'cancelled': 0
            }
            
            base_progress = progress_map.get(status, 0)
            
            # Look for training progress in events
            training_progress = 0
            for event in events:
                message = event.get('message', '').lower()
                if 'step' in message and 'of' in message:
                    try:
                        # Extract step information like "Step 50 of 100"
                        parts = message.split()
                        step_idx = parts.index('step') + 1
                        of_idx = parts.index('of') + 1
                        
                        current_step = int(parts[step_idx])
                        total_steps = int(parts[of_idx])
                        
                        training_progress = (current_step / total_steps) * 100
                        break
                    except (ValueError, IndexError):
                        continue
            
            # Combine base progress with training progress for running jobs
            if status == 'running' and training_progress > 0:
                final_progress = 20 + (training_progress * 0.7)  # 20% base + 70% for training
            else:
                final_progress = base_progress
            
            return {
                'percentage': min(final_progress, 100),
                'status_description': self._get_status_description(status),
                'estimated_completion': self._estimate_completion_time(status, events)
            }
            
        except Exception as e:
            logger.warning(f"Error estimating job progress: {str(e)}")
            return {
                'percentage': 0,
                'status_description': status,
                'estimated_completion': None
            }
    
    def _get_status_description(self, status: str) -> str:
        """Get human-readable description for job status."""
        descriptions = {
            'validating_files': 'Validating uploaded training files',
            'queued': 'Job queued and waiting to start',
            'running': 'Training in progress',
            'succeeded': 'Training completed successfully',
            'failed': 'Training failed',
            'cancelled': 'Training was cancelled'
        }
        return descriptions.get(status, f'Unknown status: {status}')
    
    def _estimate_completion_time(self, status: str, events: List[Dict[str, Any]]) -> Optional[str]:
        """Estimate completion time based on job status and events."""
        try:
            if status in ['succeeded', 'failed', 'cancelled']:
                return None
            
            if status == 'queued':
                return "Waiting in queue - completion time depends on queue length"
            
            if status == 'running':
                # Look for timing information in events
                for event in events:
                    message = event.get('message', '').lower()
                    if 'eta' in message or 'estimated' in message:
                        return "Check recent events for timing estimates"
                
                return "Training in progress - completion time varies by dataset size"
            
            return "Completion time unknown"
            
        except Exception:
            return None
    
    def list_fine_tuning_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent fine-tuning jobs.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of job information dictionaries
        """
        try:
            if not self.client:
                return []
            
            jobs_response = self.client.fine_tuning.jobs.list(limit=limit)
            
            jobs_info = []
            for job in jobs_response.data:
                job_info = {
                    'job_id': job.id,
                    'status': job.status,
                    'model': job.model,
                    'created_at': job.created_at,
                    'finished_at': job.finished_at,
                    'fine_tuned_model': job.fine_tuned_model,
                    'trained_tokens': job.trained_tokens
                }
                jobs_info.append(job_info)
            
            return jobs_info
            
        except Exception as e:
            logger.error(f"Error listing fine-tuning jobs: {str(e)}")
            return []
    
    def cancel_fine_tuning_job(self, job_id: str) -> Tuple[bool, str]:
        """
        Cancel a running fine-tuning job.
        
        Args:
            job_id: The fine-tuning job ID to cancel
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not self.client:
                return False, "OpenAI client not initialized"
            
            # Cancel the job
            cancelled_job = self.client.fine_tuning.jobs.cancel(job_id)
            
            # Update stored job status
            self._update_fine_tuning_job_status(job_id, cancelled_job.status)
            
            logger.info(f"Fine-tuning job {job_id} cancelled")
            return True, f"Job {job_id} cancelled successfully"
            
        except openai.NotFoundError:
            return False, f"Fine-tuning job {job_id} not found"
            
        except Exception as e:
            error_msg = f"Error cancelling fine-tuning job: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def deploy_fine_tuned_model(self, job_id: str) -> Tuple[bool, str]:
        """
        Deploy a successfully fine-tuned model for use.
        
        Args:
            job_id: The fine-tuning job ID
            
        Returns:
            Tuple of (success, model_name or error_message)
        """
        try:
            # Get job status
            job_status = self.monitor_fine_tuning_job(job_id)
            
            if 'error' in job_status:
                return False, job_status['error']
            
            if job_status['status'] != 'succeeded':
                return False, f"Job not completed successfully. Status: {job_status['status']}"
            
            fine_tuned_model = job_status.get('fine_tuned_model')
            if not fine_tuned_model:
                return False, "No fine-tuned model available from completed job"
            
            # Update job record with final model name
            self._update_fine_tuning_job_status(job_id, 'succeeded', fine_tuned_model)
            
            logger.info(f"Fine-tuned model {fine_tuned_model} ready for deployment")
            return True, fine_tuned_model
            
        except Exception as e:
            error_msg = f"Error deploying fine-tuned model: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def compare_model_performance(self, base_model_results: Dict[str, float],
                                fine_tuned_model_results: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare performance between base model and fine-tuned model.
        
        Args:
            base_model_results: Performance metrics for base model
            fine_tuned_model_results: Performance metrics for fine-tuned model
            
        Returns:
            Dictionary with comparison results and recommendations
        """
        try:
            comparison = {
                'base_model': base_model_results,
                'fine_tuned_model': fine_tuned_model_results,
                'improvements': {},
                'degradations': {},
                'recommendation': 'insufficient_data'
            }
            
            # Calculate improvements and degradations
            for metric, base_value in base_model_results.items():
                if metric in fine_tuned_model_results:
                    fine_tuned_value = fine_tuned_model_results[metric]
                    difference = fine_tuned_value - base_value
                    percentage_change = (difference / base_value) * 100 if base_value != 0 else 0
                    
                    if difference > 0:
                        comparison['improvements'][metric] = {
                            'absolute_change': difference,
                            'percentage_change': percentage_change
                        }
                    elif difference < 0:
                        comparison['degradations'][metric] = {
                            'absolute_change': abs(difference),
                            'percentage_change': abs(percentage_change)
                        }
            
            # Generate recommendation
            total_improvements = len(comparison['improvements'])
            total_degradations = len(comparison['degradations'])
            
            if total_improvements > total_degradations:
                comparison['recommendation'] = 'use_fine_tuned'
                comparison['reason'] = f"Fine-tuned model shows improvements in {total_improvements} metrics vs {total_degradations} degradations"
            elif total_degradations > total_improvements:
                comparison['recommendation'] = 'use_base_model'
                comparison['reason'] = f"Base model performs better with {total_degradations} metrics degraded vs {total_improvements} improved"
            else:
                comparison['recommendation'] = 'equivalent_performance'
                comparison['reason'] = "Models show equivalent performance - consider cost and latency factors"
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing model performance: {str(e)}")
            return {'error': str(e)}
    
    def _save_fine_tuning_job(self, job_info: FineTuningJob):
        """Save fine-tuning job information to storage."""
        try:
            jobs_file = os.path.join(self.training_data_dir, "fine_tuning_jobs.json")
            
            # Load existing jobs
            existing_jobs = []
            if os.path.exists(jobs_file):
                with open(jobs_file, 'r', encoding='utf-8') as f:
                    existing_jobs = json.load(f)
            
            # Add new job
            job_data = {
                'job_id': job_info.job_id,
                'model_name': job_info.model_name,
                'training_file_id': job_info.training_file_id,
                'status': job_info.status,
                'created_at': job_info.created_at.isoformat(),
                'completed_at': job_info.completed_at.isoformat() if job_info.completed_at else None,
                'fine_tuned_model': job_info.fine_tuned_model
            }
            
            existing_jobs.append(job_data)
            
            # Save updated jobs
            with open(jobs_file, 'w', encoding='utf-8') as f:
                json.dump(existing_jobs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved fine-tuning job info: {job_info.job_id}")
            
        except Exception as e:
            logger.error(f"Error saving fine-tuning job: {str(e)}")
    
    def _update_fine_tuning_job_status(self, job_id: str, status: str, 
                                     fine_tuned_model: Optional[str] = None):
        """Update the status of a stored fine-tuning job."""
        try:
            jobs_file = os.path.join(self.training_data_dir, "fine_tuning_jobs.json")
            
            if not os.path.exists(jobs_file):
                return
            
            # Load existing jobs
            with open(jobs_file, 'r', encoding='utf-8') as f:
                jobs = json.load(f)
            
            # Update job status
            for job in jobs:
                if job['job_id'] == job_id:
                    job['status'] = status
                    if fine_tuned_model:
                        job['fine_tuned_model'] = fine_tuned_model
                    if status in ['succeeded', 'failed', 'cancelled']:
                        job['completed_at'] = datetime.now().isoformat()
                    break
            
            # Save updated jobs
            with open(jobs_file, 'w', encoding='utf-8') as f:
                json.dump(jobs, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error updating fine-tuning job status: {str(e)}")
    
    def get_stored_fine_tuning_jobs(self) -> List[Dict[str, Any]]:
        """Get locally stored fine-tuning job information."""
        try:
            jobs_file = os.path.join(self.training_data_dir, "fine_tuning_jobs.json")
            
            if not os.path.exists(jobs_file):
                return []
            
            with open(jobs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
            
        except Exception as e:
            logger.error(f"Error getting stored fine-tuning jobs: {str(e)}")
            return []