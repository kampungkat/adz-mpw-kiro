"""
Unit tests for ModelTrainingManager.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from business_logic.model_training_manager import ModelTrainingManager
from models.data_models import ClientBrief, MediaPlan, FormatAllocation, TrainingData


class TestModelTrainingManager(unittest.TestCase):
    """Test cases for ModelTrainingManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test manager with mocked OpenAI client
        with patch('business_logic.model_training_manager.config_manager') as mock_config:
            mock_config.get_openai_api_key.return_value = "test-api-key"
            self.manager = ModelTrainingManager(skip_openai_init=True)
        
        # Override training data directory to use temp directory
        self.manager.training_data_dir = self.temp_dir
        self.manager.training_data_file = os.path.join(self.temp_dir, "test_data.json")
        self.manager.openai_format_file = os.path.join(self.temp_dir, "test_openai.jsonl")
        
        # Clear any existing training data to ensure clean state
        self.manager.training_data = []
        
        # Create test data
        self.test_client_brief = ClientBrief(
            brand_name="Test Brand",
            budget=50000.0,
            country="US",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="AI",
            selected_formats=None
        )
        
        self.test_allocation = FormatAllocation(
            format_name="Display Banner",
            budget_allocation=25000.0,
            cpm=5.0,
            estimated_impressions=5000000,
            recommended_sites=["site1.com", "site2.com"],
            notes="High-impact placement"
        )
        
        self.test_media_plan = MediaPlan(
            plan_id="test_plan_1",
            title="Reach-Focused Strategy",
            total_budget=50000.0,
            allocations=[self.test_allocation],
            estimated_reach=2000000,
            estimated_impressions=5000000,
            rationale="Maximize reach with cost-effective placements",
            created_at=datetime.now()
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ModelTrainingManager initialization."""
        self.assertIsInstance(self.manager, ModelTrainingManager)
        self.assertEqual(self.manager.min_training_examples, 10)
        self.assertEqual(self.manager.validation_split, 0.1)
        self.assertIsInstance(self.manager.training_data, list)
    
    def test_format_campaign_brief(self):
        """Test campaign brief formatting."""
        formatted_brief = self.manager._format_campaign_brief(self.test_client_brief)
        
        self.assertIn("Brand: Test Brand", formatted_brief)
        self.assertIn("Budget: $50,000.00", formatted_brief)
        self.assertIn("Market: US", formatted_brief)
        self.assertIn("Campaign Period: Q1 2024", formatted_brief)
        self.assertIn("Objective: Brand Awareness", formatted_brief)
        self.assertIn("Planning Mode: AI", formatted_brief)
    
    def test_format_campaign_brief_with_selected_formats(self):
        """Test campaign brief formatting with selected formats."""
        brief_with_formats = ClientBrief(
            brand_name="Test Brand",
            budget=50000.0,
            country="US",
            campaign_period="Q1 2024",
            objective="Brand Awareness",
            planning_mode="Manual",
            selected_formats=["Display Banner", "Video Pre-roll"]
        )
        
        formatted_brief = self.manager._format_campaign_brief(brief_with_formats)
        self.assertIn("Selected Formats: Display Banner, Video Pre-roll", formatted_brief)
    
    def test_format_generated_plan(self):
        """Test generated plan formatting."""
        formatted_plan = self.manager._format_generated_plan(self.test_media_plan)
        
        self.assertIn("Title: Reach-Focused Strategy", formatted_plan)
        self.assertIn("Rationale: Maximize reach with cost-effective placements", formatted_plan)
        self.assertIn("Total Budget: $50,000.00", formatted_plan)
        self.assertIn("Estimated Reach: 2,000,000", formatted_plan)
        self.assertIn("Estimated Impressions: 5,000,000", formatted_plan)
        self.assertIn("Allocations:", formatted_plan)
        self.assertIn("Format: Display Banner", formatted_plan)
        self.assertIn("Budget: $25,000.00", formatted_plan)
        self.assertIn("CPM: $5.00", formatted_plan)
        self.assertIn("Sites: site1.com, site2.com", formatted_plan)
    
    def test_validate_training_data_quality_valid(self):
        """Test training data quality validation with valid data."""
        training_data = TrainingData(
            campaign_brief=self.manager._format_campaign_brief(self.test_client_brief),
            generated_plan=self.manager._format_generated_plan(self.test_media_plan),
            performance_metrics=None,
            created_at=datetime.now(),
            validated=False
        )
        
        is_valid = self.manager._validate_training_data_quality(training_data)
        self.assertTrue(is_valid)
    
    def test_validate_training_data_quality_invalid_brief(self):
        """Test training data quality validation with invalid brief."""
        training_data = TrainingData(
            campaign_brief="Too short",  # Too short
            generated_plan=self.manager._format_generated_plan(self.test_media_plan),
            performance_metrics=None,
            created_at=datetime.now(),
            validated=False
        )
        
        is_valid = self.manager._validate_training_data_quality(training_data)
        self.assertFalse(is_valid)
    
    def test_validate_training_data_quality_invalid_plan(self):
        """Test training data quality validation with invalid plan."""
        training_data = TrainingData(
            campaign_brief=self.manager._format_campaign_brief(self.test_client_brief),
            generated_plan="Too short",  # Too short
            performance_metrics=None,
            created_at=datetime.now(),
            validated=False
        )
        
        is_valid = self.manager._validate_training_data_quality(training_data)
        self.assertFalse(is_valid)
    
    def test_validate_training_data_quality_missing_fields(self):
        """Test training data quality validation with missing required fields."""
        training_data = TrainingData(
            campaign_brief="Some text without required fields like Brand and Budget",
            generated_plan=self.manager._format_generated_plan(self.test_media_plan),
            performance_metrics=None,
            created_at=datetime.now(),
            validated=False
        )
        
        is_valid = self.manager._validate_training_data_quality(training_data)
        self.assertFalse(is_valid)
    
    def test_collect_training_data_success(self):
        """Test successful training data collection."""
        success = self.manager.collect_training_data(
            self.test_client_brief,
            self.test_media_plan,
            performance_metrics={"ctr": 0.05, "conversion_rate": 0.02},
            validated=True
        )
        
        self.assertTrue(success)
        self.assertEqual(len(self.manager.training_data), 1)
        
        collected_data = self.manager.training_data[0]
        self.assertIn("Test Brand", collected_data.campaign_brief)
        self.assertIn("Reach-Focused Strategy", collected_data.generated_plan)
        self.assertEqual(collected_data.performance_metrics["ctr"], 0.05)
        self.assertTrue(collected_data.validated)
    
    def test_collect_training_data_duplicate(self):
        """Test that duplicate training data is rejected."""
        # Add first entry
        success1 = self.manager.collect_training_data(
            self.test_client_brief,
            self.test_media_plan
        )
        self.assertTrue(success1)
        
        # Try to add duplicate
        success2 = self.manager.collect_training_data(
            self.test_client_brief,
            self.test_media_plan
        )
        self.assertFalse(success2)
        self.assertEqual(len(self.manager.training_data), 1)
    
    def test_save_and_load_training_data(self):
        """Test saving and loading training data."""
        # Collect some data
        self.manager.collect_training_data(
            self.test_client_brief,
            self.test_media_plan,
            validated=True
        )
        
        # Create new manager instance to test loading
        with patch('business_logic.model_training_manager.config_manager') as mock_config:
            mock_config.get_openai_api_key.return_value = "test-api-key"
            new_manager = ModelTrainingManager(skip_openai_init=True)
        
        new_manager.training_data_dir = self.temp_dir
        new_manager.training_data_file = os.path.join(self.temp_dir, "test_data.json")
        new_manager._load_existing_training_data()
        
        self.assertEqual(len(new_manager.training_data), 1)
        self.assertIn("Test Brand", new_manager.training_data[0].campaign_brief)
    
    def test_convert_to_openai_format(self):
        """Test conversion to OpenAI format."""
        training_data = TrainingData(
            campaign_brief=self.manager._format_campaign_brief(self.test_client_brief),
            generated_plan=self.manager._format_generated_plan(self.test_media_plan),
            performance_metrics=None,
            created_at=datetime.now(),
            validated=True
        )
        
        openai_example = self.manager._convert_to_openai_format(training_data)
        
        self.assertIsNotNone(openai_example)
        self.assertIn("messages", openai_example)
        self.assertEqual(len(openai_example["messages"]), 3)
        
        messages = openai_example["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        
        self.assertIn("Test Brand", messages[1]["content"])
        self.assertIn("Reach-Focused Strategy", messages[2]["content"])
    
    def test_validate_openai_format_valid(self):
        """Test OpenAI format validation with valid data."""
        valid_example = {
            "messages": [
                {"role": "system", "content": "You are an expert media planner."},
                {"role": "user", "content": "Create a media plan for Brand: Test, Budget: $50000"},
                {"role": "assistant", "content": "Title: Test Plan\nRationale: Strategic approach"}
            ]
        }
        
        is_valid = self.manager._validate_openai_format(valid_example)
        self.assertTrue(is_valid)
    
    def test_validate_openai_format_invalid_structure(self):
        """Test OpenAI format validation with invalid structure."""
        invalid_examples = [
            {},  # Missing messages
            {"messages": []},  # Empty messages
            {"messages": [{"role": "user", "content": "test"}]},  # Wrong number of messages
            {"messages": [
                {"role": "wrong", "content": "test"},
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "test"}
            ]},  # Wrong role
            {"messages": [
                {"role": "system", "content": ""},  # Empty content
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "test"}
            ]}
        ]
        
        for invalid_example in invalid_examples:
            is_valid = self.manager._validate_openai_format(invalid_example)
            self.assertFalse(is_valid)
    
    def test_export_training_data_for_openai_insufficient_data(self):
        """Test export with insufficient training data."""
        # Add only a few examples (less than minimum)
        for i in range(5):
            brief = ClientBrief(
                brand_name=f"Brand {i}",
                budget=50000.0,
                country="US",
                campaign_period="Q1 2024",
                objective="Brand Awareness",
                planning_mode="AI"
            )
            self.manager.collect_training_data(brief, self.test_media_plan)
        
        success, message = self.manager.export_training_data_for_openai()
        self.assertFalse(success)
        self.assertIn("Insufficient training examples", message)
    
    def test_export_training_data_for_openai_success(self):
        """Test successful export of training data."""
        # Add sufficient training examples
        for i in range(15):
            brief = ClientBrief(
                brand_name=f"Brand {i}",
                budget=50000.0 + i * 1000,
                country="US",
                campaign_period="Q1 2024",
                objective="Brand Awareness",
                planning_mode="AI"
            )
            self.manager.collect_training_data(brief, self.test_media_plan, validated=True)
        
        success, file_path = self.manager.export_training_data_for_openai()
        self.assertTrue(success)
        self.assertTrue(os.path.exists(file_path))
        
        # Verify JSONL format
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 15)
            
            # Check first line is valid JSON
            first_example = json.loads(lines[0])
            self.assertIn("messages", first_example)
            self.assertEqual(len(first_example["messages"]), 3)
    
    def test_validate_training_data_requirements(self):
        """Test training data requirements validation."""
        # Test with insufficient data
        validation_result = self.manager.validate_training_data_requirements()
        self.assertFalse(validation_result["ready_for_fine_tuning"])
        self.assertIn("Need at least", validation_result["recommendations"][0])
        
        # Add sufficient data
        for i in range(12):
            brief = ClientBrief(
                brand_name=f"Brand {i}",
                budget=50000.0,
                country="US",
                campaign_period="Q1 2024",
                objective="Brand Awareness",
                planning_mode="AI"
            )
            self.manager.collect_training_data(brief, self.test_media_plan, validated=True)
        
        validation_result = self.manager.validate_training_data_requirements()
        self.assertTrue(validation_result["ready_for_fine_tuning"])
        self.assertEqual(validation_result["total_examples"], 12)
        self.assertEqual(validation_result["validated_examples"], 12)
    
    def test_get_training_data_summary_empty(self):
        """Test training data summary with no data."""
        summary = self.manager.get_training_data_summary()
        
        self.assertEqual(summary["total_examples"], 0)
        self.assertEqual(summary["validated_examples"], 0)
        self.assertIsNone(summary["date_range"])
        self.assertFalse(summary["ready_for_export"])
    
    def test_get_training_data_summary_with_data(self):
        """Test training data summary with data."""
        # Add some training data
        for i in range(5):
            brief = ClientBrief(
                brand_name=f"Brand {i}",
                budget=50000.0,
                country="US",
                campaign_period="Q1 2024",
                objective="Brand Awareness",
                planning_mode="AI"
            )
            self.manager.collect_training_data(brief, self.test_media_plan, validated=i < 3)
        
        summary = self.manager.get_training_data_summary()
        
        self.assertEqual(summary["total_examples"], 5)
        self.assertEqual(summary["validated_examples"], 3)
        self.assertEqual(summary["unvalidated_examples"], 2)
        self.assertIsNotNone(summary["date_range"])
        self.assertFalse(summary["ready_for_export"])  # Still less than minimum
    
    def test_mark_training_data_validated(self):
        """Test marking training data as validated."""
        # Add training data
        self.manager.collect_training_data(
            self.test_client_brief,
            self.test_media_plan,
            validated=False
        )
        
        # Mark as validated
        brief_text = self.manager._format_campaign_brief(self.test_client_brief)
        plan_text = self.manager._format_generated_plan(self.test_media_plan)
        
        success = self.manager.mark_training_data_validated(
            brief_text,
            plan_text,
            performance_metrics={"ctr": 0.05}
        )
        
        self.assertTrue(success)
        self.assertTrue(self.manager.training_data[0].validated)
        self.assertEqual(self.manager.training_data[0].performance_metrics["ctr"], 0.05)
    
    def test_mark_training_data_validated_not_found(self):
        """Test marking non-existent training data as validated."""
        success = self.manager.mark_training_data_validated(
            "Non-existent brief",
            "Non-existent plan"
        )
        
        self.assertFalse(success)
    
    def test_clear_training_data(self):
        """Test clearing training data."""
        # Add some data
        self.manager.collect_training_data(self.test_client_brief, self.test_media_plan)
        self.assertEqual(len(self.manager.training_data), 1)
        
        # Try to clear without confirmation
        success = self.manager.clear_training_data(confirmed=False)
        self.assertFalse(success)
        self.assertEqual(len(self.manager.training_data), 1)
        
        # Clear with confirmation
        success = self.manager.clear_training_data(confirmed=True)
        self.assertTrue(success)
        self.assertEqual(len(self.manager.training_data), 0)


    def test_initiate_fine_tuning_job_no_client(self):
        """Test fine-tuning job initiation without OpenAI client."""
        success, message = self.manager.initiate_fine_tuning_job()
        self.assertFalse(success)
        self.assertIn("OpenAI client not initialized", message)
    
    @patch('business_logic.model_training_manager.OpenAI')
    def test_initiate_fine_tuning_job_success(self, mock_openai_class):
        """Test successful fine-tuning job initiation."""
        # Mock OpenAI client and responses
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        self.manager.client = mock_client
        
        # Mock file upload
        mock_file = Mock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file
        
        # Mock fine-tuning job creation
        mock_job = Mock()
        mock_job.id = "ft-job-123"
        mock_job.status = "validating_files"
        mock_client.fine_tuning.jobs.create.return_value = mock_job
        
        # Add sufficient training data
        for i in range(15):
            brief = ClientBrief(
                brand_name=f"Brand {i}",
                budget=50000.0,
                country="US",
                campaign_period="Q1 2024",
                objective="Brand Awareness",
                planning_mode="AI"
            )
            self.manager.collect_training_data(brief, self.test_media_plan, validated=True)
        
        success, job_id = self.manager.initiate_fine_tuning_job()
        
        self.assertTrue(success)
        self.assertEqual(job_id, "ft-job-123")
        mock_client.files.create.assert_called_once()
        mock_client.fine_tuning.jobs.create.assert_called_once()
    
    @patch('business_logic.model_training_manager.OpenAI')
    def test_monitor_fine_tuning_job_success(self, mock_openai_class):
        """Test successful fine-tuning job monitoring."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        self.manager.client = mock_client
        
        # Mock job retrieval
        mock_job = Mock()
        mock_job.id = "ft-job-123"
        mock_job.status = "running"
        mock_job.model = "gpt-3.5-turbo"
        mock_job.created_at = 1234567890
        mock_job.finished_at = None
        mock_job.fine_tuned_model = None
        mock_job.training_file = "file-123"
        mock_job.validation_file = None
        mock_job.hyperparameters = {}
        mock_job.result_files = []
        mock_job.trained_tokens = 1000
        mock_job.error = None
        
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_job
        
        # Mock events
        mock_events = Mock()
        mock_events.data = []
        mock_client.fine_tuning.jobs.list_events.return_value = mock_events
        
        result = self.manager.monitor_fine_tuning_job("ft-job-123")
        
        self.assertEqual(result['job_id'], "ft-job-123")
        self.assertEqual(result['status'], "running")
        self.assertIn('progress_estimate', result)
        mock_client.fine_tuning.jobs.retrieve.assert_called_once_with("ft-job-123")
    
    def test_monitor_fine_tuning_job_no_client(self):
        """Test fine-tuning job monitoring without OpenAI client."""
        result = self.manager.monitor_fine_tuning_job("ft-job-123")
        self.assertIn('error', result)
        self.assertIn('OpenAI client not initialized', result['error'])
    
    def test_estimate_job_progress(self):
        """Test job progress estimation."""
        # Test different statuses
        test_cases = [
            ('validating_files', [], 10),
            ('queued', [], 20),
            ('running', [], 50),
            ('succeeded', [], 100),
            ('failed', [], 0)
        ]
        
        for status, events, expected_min in test_cases:
            progress = self.manager._estimate_job_progress(status, events)
            self.assertGreaterEqual(progress['percentage'], expected_min)
            self.assertIn('status_description', progress)
    
    def test_get_status_description(self):
        """Test status description generation."""
        descriptions = {
            'validating_files': 'Validating uploaded training files',
            'queued': 'Job queued and waiting to start',
            'running': 'Training in progress',
            'succeeded': 'Training completed successfully',
            'failed': 'Training failed',
            'cancelled': 'Training was cancelled'
        }
        
        for status, expected_desc in descriptions.items():
            desc = self.manager._get_status_description(status)
            self.assertEqual(desc, expected_desc)
    
    @patch('business_logic.model_training_manager.OpenAI')
    def test_list_fine_tuning_jobs(self, mock_openai_class):
        """Test listing fine-tuning jobs."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        self.manager.client = mock_client
        
        # Mock jobs response
        mock_job1 = Mock()
        mock_job1.id = "ft-job-1"
        mock_job1.status = "succeeded"
        mock_job1.model = "gpt-3.5-turbo"
        mock_job1.created_at = 1234567890
        mock_job1.finished_at = 1234567900
        mock_job1.fine_tuned_model = "ft:gpt-3.5-turbo:model1"
        mock_job1.trained_tokens = 1000
        
        mock_jobs_response = Mock()
        mock_jobs_response.data = [mock_job1]
        mock_client.fine_tuning.jobs.list.return_value = mock_jobs_response
        
        jobs = self.manager.list_fine_tuning_jobs()
        
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['job_id'], "ft-job-1")
        self.assertEqual(jobs[0]['status'], "succeeded")
        mock_client.fine_tuning.jobs.list.assert_called_once_with(limit=10)
    
    @patch('business_logic.model_training_manager.OpenAI')
    def test_cancel_fine_tuning_job(self, mock_openai_class):
        """Test cancelling a fine-tuning job."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        self.manager.client = mock_client
        
        # Mock job cancellation
        mock_cancelled_job = Mock()
        mock_cancelled_job.status = "cancelled"
        mock_client.fine_tuning.jobs.cancel.return_value = mock_cancelled_job
        
        success, message = self.manager.cancel_fine_tuning_job("ft-job-123")
        
        self.assertTrue(success)
        self.assertIn("cancelled successfully", message)
        mock_client.fine_tuning.jobs.cancel.assert_called_once_with("ft-job-123")
    
    def test_deploy_fine_tuned_model_not_completed(self):
        """Test deploying a model that hasn't completed training."""
        # Mock monitor method to return incomplete job
        with patch.object(self.manager, 'monitor_fine_tuning_job') as mock_monitor:
            mock_monitor.return_value = {
                'status': 'running',
                'fine_tuned_model': None
            }
            
            success, message = self.manager.deploy_fine_tuned_model("ft-job-123")
            
            self.assertFalse(success)
            self.assertIn("not completed successfully", message)
    
    def test_deploy_fine_tuned_model_success(self):
        """Test successful model deployment."""
        # Mock monitor method to return completed job
        with patch.object(self.manager, 'monitor_fine_tuning_job') as mock_monitor:
            mock_monitor.return_value = {
                'status': 'succeeded',
                'fine_tuned_model': 'ft:gpt-3.5-turbo:model1'
            }
            
            success, model_name = self.manager.deploy_fine_tuned_model("ft-job-123")
            
            self.assertTrue(success)
            self.assertEqual(model_name, 'ft:gpt-3.5-turbo:model1')
    
    def test_compare_model_performance(self):
        """Test model performance comparison."""
        base_results = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75
        }
        
        fine_tuned_results = {
            'accuracy': 0.90,  # Improved
            'precision': 0.78,  # Degraded
            'recall': 0.82     # Improved
        }
        
        comparison = self.manager.compare_model_performance(base_results, fine_tuned_results)
        
        self.assertEqual(len(comparison['improvements']), 2)  # accuracy, recall
        self.assertEqual(len(comparison['degradations']), 1)  # precision
        self.assertEqual(comparison['recommendation'], 'use_fine_tuned')
        self.assertIn('reason', comparison)
    
    def test_compare_model_performance_equivalent(self):
        """Test model performance comparison with equivalent results."""
        base_results = {
            'accuracy': 0.85,
            'precision': 0.80
        }
        
        fine_tuned_results = {
            'accuracy': 0.90,  # Improved
            'precision': 0.75   # Degraded
        }
        
        comparison = self.manager.compare_model_performance(base_results, fine_tuned_results)
        
        self.assertEqual(len(comparison['improvements']), 1)
        self.assertEqual(len(comparison['degradations']), 1)
        self.assertEqual(comparison['recommendation'], 'equivalent_performance')
    
    def test_save_and_get_stored_fine_tuning_jobs(self):
        """Test saving and retrieving fine-tuning job information."""
        from models.data_models import FineTuningJob
        
        job_info = FineTuningJob(
            job_id="ft-job-123",
            model_name="gpt-3.5-turbo",
            training_file_id="file-123",
            status="running",
            created_at=datetime.now(),
            completed_at=None,
            fine_tuned_model=None
        )
        
        # Save job info
        self.manager._save_fine_tuning_job(job_info)
        
        # Retrieve stored jobs
        stored_jobs = self.manager.get_stored_fine_tuning_jobs()
        
        self.assertEqual(len(stored_jobs), 1)
        self.assertEqual(stored_jobs[0]['job_id'], "ft-job-123")
        self.assertEqual(stored_jobs[0]['status'], "running")
    
    def test_update_fine_tuning_job_status(self):
        """Test updating fine-tuning job status."""
        from models.data_models import FineTuningJob
        
        # Create and save initial job
        job_info = FineTuningJob(
            job_id="ft-job-123",
            model_name="gpt-3.5-turbo",
            training_file_id="file-123",
            status="running",
            created_at=datetime.now(),
            completed_at=None,
            fine_tuned_model=None
        )
        
        self.manager._save_fine_tuning_job(job_info)
        
        # Update status
        self.manager._update_fine_tuning_job_status(
            "ft-job-123", 
            "succeeded", 
            "ft:gpt-3.5-turbo:model1"
        )
        
        # Check updated status
        stored_jobs = self.manager.get_stored_fine_tuning_jobs()
        updated_job = stored_jobs[0]
        
        self.assertEqual(updated_job['status'], "succeeded")
        self.assertEqual(updated_job['fine_tuned_model'], "ft:gpt-3.5-turbo:model1")
        self.assertIsNotNone(updated_job['completed_at'])


if __name__ == '__main__':
    unittest.main()