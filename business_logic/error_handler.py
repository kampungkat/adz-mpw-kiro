"""
Comprehensive error handling and user feedback system.

This module provides centralized error handling, retry mechanisms,
and user-friendly feedback for all system components.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import openai
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification."""
    API_ERROR = "api_error"
    DATA_ERROR = "data_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    SYSTEM_ERROR = "system_error"
    USER_ERROR = "user_error"


@dataclass
class ErrorInfo:
    """Structured error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    technical_details: Optional[str] = None
    suggested_action: Optional[str] = None
    retry_possible: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 exponential_backoff: bool = True, max_delay: float = 60.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.exponential_backoff = exponential_backoff
        self.max_delay = max_delay


class ErrorHandler:
    """
    Centralized error handling and user feedback system.
    
    Provides retry mechanisms, user-friendly error messages,
    and fallback strategies for various error scenarios.
    """
    
    def __init__(self):
        self.error_history = []
        self.rate_limit_tracker = {}
        
    def handle_openai_error(self, error: Exception, context: str = "") -> ErrorInfo:
        """
        Handle OpenAI API specific errors with appropriate user feedback.
        
        Args:
            error: The OpenAI exception
            context: Additional context about the operation
            
        Returns:
            ErrorInfo object with structured error information
        """
        try:
            if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
                status_code = error.response.status_code
                
                if status_code == 401:
                    return ErrorInfo(
                        category=ErrorCategory.API_ERROR,
                        severity=ErrorSeverity.CRITICAL,
                        message=f"OpenAI API authentication failed: {str(error)}",
                        user_message="API authentication failed. Please check your OpenAI API key configuration.",
                        suggested_action="Verify your OpenAI API key in the system settings.",
                        retry_possible=False
                    )
                
                elif status_code == 429:
                    # Rate limiting
                    self._track_rate_limit()
                    return ErrorInfo(
                        category=ErrorCategory.API_ERROR,
                        severity=ErrorSeverity.WARNING,
                        message=f"OpenAI API rate limit exceeded: {str(error)}",
                        user_message="API rate limit exceeded. The system will automatically retry in a moment.",
                        suggested_action="Please wait a moment and try again. Consider upgrading your OpenAI plan for higher limits.",
                        retry_possible=True
                    )
                
                elif status_code == 503:
                    return ErrorInfo(
                        category=ErrorCategory.API_ERROR,
                        severity=ErrorSeverity.ERROR,
                        message=f"OpenAI API service unavailable: {str(error)}",
                        user_message="The AI service is temporarily unavailable. Please try again in a few minutes.",
                        suggested_action="Wait a few minutes and retry. If the problem persists, check OpenAI's status page.",
                        retry_possible=True
                    )
                
                elif status_code >= 500:
                    return ErrorInfo(
                        category=ErrorCategory.API_ERROR,
                        severity=ErrorSeverity.ERROR,
                        message=f"OpenAI API server error: {str(error)}",
                        user_message="The AI service encountered an internal error. Please try again.",
                        suggested_action="Retry the operation. If the problem persists, contact support.",
                        retry_possible=True
                    )
            
            # Handle specific OpenAI exception types
            if "timeout" in str(error).lower():
                return ErrorInfo(
                    category=ErrorCategory.NETWORK_ERROR,
                    severity=ErrorSeverity.WARNING,
                    message=f"OpenAI API timeout: {str(error)}",
                    user_message="The AI service request timed out. Please try again.",
                    suggested_action="Check your internet connection and retry.",
                    retry_possible=True
                )
            
            # Generic OpenAI error
            return ErrorInfo(
                category=ErrorCategory.API_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"OpenAI API error in {context}: {str(error)}",
                user_message="An error occurred while communicating with the AI service.",
                technical_details=str(error),
                suggested_action="Please try again. If the problem persists, contact support.",
                retry_possible=True
            )
            
        except Exception as e:
            logger.error(f"Error in error handler: {str(e)}")
            return ErrorInfo(
                category=ErrorCategory.SYSTEM_ERROR,
                severity=ErrorSeverity.CRITICAL,
                message=f"Error handler failure: {str(e)}",
                user_message="A system error occurred. Please contact support.",
                retry_possible=False
            )
    
    def handle_data_error(self, error: Exception, context: str = "") -> ErrorInfo:
        """
        Handle data-related errors (file not found, parsing errors, etc.).
        
        Args:
            error: The data exception
            context: Additional context about the operation
            
        Returns:
            ErrorInfo object with structured error information
        """
        error_str = str(error).lower()
        
        if "file not found" in error_str or "no such file" in error_str:
            return ErrorInfo(
                category=ErrorCategory.DATA_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Required data file not found: {str(error)}",
                user_message="Required data files are missing. Please upload the latest rate cards and site lists.",
                suggested_action="Upload the required Excel files (rate cards and site lists) in the data management section.",
                retry_possible=False
            )
        
        elif "permission denied" in error_str:
            return ErrorInfo(
                category=ErrorCategory.DATA_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"File permission error: {str(error)}",
                user_message="Cannot access data files due to permission restrictions.",
                suggested_action="Check file permissions or contact your system administrator.",
                retry_possible=False
            )
        
        elif "corrupt" in error_str or "invalid" in error_str or "malformed" in error_str:
            return ErrorInfo(
                category=ErrorCategory.DATA_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Data file corruption detected: {str(error)}",
                user_message="The data files appear to be corrupted or in an invalid format.",
                suggested_action="Re-upload the data files ensuring they are in the correct Excel format.",
                retry_possible=False
            )
        
        elif "sheet" in error_str and "not found" in error_str:
            return ErrorInfo(
                category=ErrorCategory.DATA_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Required Excel sheet missing: {str(error)}",
                user_message="The uploaded Excel file is missing required sheets (APX - Impact, APX - Reach, or market sheets).",
                suggested_action="Ensure your Excel files contain all required sheets with the correct names.",
                retry_possible=False
            )
        
        # Generic data error
        return ErrorInfo(
            category=ErrorCategory.DATA_ERROR,
            severity=ErrorSeverity.ERROR,
            message=f"Data processing error in {context}: {str(error)}",
            user_message="An error occurred while processing data files.",
            technical_details=str(error),
            suggested_action="Check your data files and try again. Contact support if the problem persists.",
            retry_possible=False
        )
    
    def handle_network_error(self, error: Exception, context: str = "") -> ErrorInfo:
        """
        Handle network-related errors.
        
        Args:
            error: The network exception
            context: Additional context about the operation
            
        Returns:
            ErrorInfo object with structured error information
        """
        error_str = str(error).lower()
        
        if "connection" in error_str and ("refused" in error_str or "failed" in error_str):
            return ErrorInfo(
                category=ErrorCategory.NETWORK_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Network connection failed: {str(error)}",
                user_message="Cannot connect to external services. Please check your internet connection.",
                suggested_action="Check your internet connection and firewall settings, then try again.",
                retry_possible=True
            )
        
        elif "timeout" in error_str:
            return ErrorInfo(
                category=ErrorCategory.NETWORK_ERROR,
                severity=ErrorSeverity.WARNING,
                message=f"Network timeout: {str(error)}",
                user_message="The request timed out. This may be due to slow internet or high server load.",
                suggested_action="Check your internet connection and try again.",
                retry_possible=True
            )
        
        elif "dns" in error_str or "resolve" in error_str:
            return ErrorInfo(
                category=ErrorCategory.NETWORK_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"DNS resolution failed: {str(error)}",
                user_message="Cannot resolve external service addresses. Please check your DNS settings.",
                suggested_action="Check your DNS settings or try again later.",
                retry_possible=True
            )
        
        # Generic network error
        return ErrorInfo(
            category=ErrorCategory.NETWORK_ERROR,
            severity=ErrorSeverity.ERROR,
            message=f"Network error in {context}: {str(error)}",
            user_message="A network error occurred while communicating with external services.",
            technical_details=str(error),
            suggested_action="Check your internet connection and try again.",
            retry_possible=True
        )
    
    def handle_validation_error(self, error: Exception, context: str = "") -> ErrorInfo:
        """
        Handle validation errors with specific user guidance.
        
        Args:
            error: The validation exception
            context: Additional context about the operation
            
        Returns:
            ErrorInfo object with structured error information
        """
        return ErrorInfo(
            category=ErrorCategory.VALIDATION_ERROR,
            severity=ErrorSeverity.WARNING,
            message=f"Validation error in {context}: {str(error)}",
            user_message=str(error),  # Validation errors are usually user-friendly
            suggested_action="Please correct the highlighted issues and try again.",
            retry_possible=False
        )
    
    def retry_with_backoff(self, func: Callable, config: RetryConfig = None, 
                          context: str = "") -> Tuple[bool, Any, Optional[ErrorInfo]]:
        """
        Execute a function with retry logic and exponential backoff.
        
        Args:
            func: Function to execute
            config: Retry configuration
            context: Context for error reporting
            
        Returns:
            Tuple of (success, result, error_info)
        """
        if config is None:
            config = RetryConfig()
        
        last_error = None
        
        for attempt in range(config.max_attempts):
            try:
                result = func()
                return True, result, None
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{config.max_attempts} failed in {context}: {str(e)}")
                
                # Don't retry on the last attempt
                if attempt == config.max_attempts - 1:
                    break
                
                # Calculate delay
                if config.exponential_backoff:
                    delay = min(config.base_delay * (2 ** attempt), config.max_delay)
                else:
                    delay = config.base_delay
                
                # Check if error is retryable
                error_info = self.classify_error(e, context)
                if not error_info.retry_possible:
                    break
                
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # All attempts failed
        error_info = self.classify_error(last_error, context)
        return False, None, error_info
    
    def classify_error(self, error: Exception, context: str = "") -> ErrorInfo:
        """
        Classify an error and return appropriate ErrorInfo.
        
        Args:
            error: The exception to classify
            context: Additional context about the operation
            
        Returns:
            ErrorInfo object with structured error information
        """
        try:
            # Check for specific error types
            if isinstance(error, (openai.APIError, openai.OpenAIError)):
                return self.handle_openai_error(error, context)
            
            elif isinstance(error, (FileNotFoundError, PermissionError, IOError)):
                return self.handle_data_error(error, context)
            
            elif isinstance(error, (ConnectionError, TimeoutError)):
                return self.handle_network_error(error, context)
            
            elif isinstance(error, (ValueError, TypeError)) and "validation" in str(error).lower():
                return self.handle_validation_error(error, context)
            
            # Check error message for classification
            error_str = str(error).lower()
            
            if any(keyword in error_str for keyword in ["openai", "api", "gpt"]):
                return self.handle_openai_error(error, context)
            
            elif any(keyword in error_str for keyword in ["file", "excel", "sheet", "data"]):
                return self.handle_data_error(error, context)
            
            elif any(keyword in error_str for keyword in ["network", "connection", "timeout", "dns"]):
                return self.handle_network_error(error, context)
            
            # Generic system error
            return ErrorInfo(
                category=ErrorCategory.SYSTEM_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Unexpected error in {context}: {str(error)}",
                user_message="An unexpected error occurred. Please try again or contact support.",
                technical_details=str(error),
                suggested_action="Try again. If the problem persists, contact support with the error details.",
                retry_possible=True
            )
            
        except Exception as e:
            logger.error(f"Error in error classification: {str(e)}")
            return ErrorInfo(
                category=ErrorCategory.SYSTEM_ERROR,
                severity=ErrorSeverity.CRITICAL,
                message=f"Critical error in error handling: {str(e)}",
                user_message="A critical system error occurred. Please contact support immediately.",
                retry_possible=False
            )
    
    def _track_rate_limit(self):
        """Track rate limiting for intelligent retry delays."""
        now = datetime.now()
        self.rate_limit_tracker['last_rate_limit'] = now
        self.rate_limit_tracker['count'] = self.rate_limit_tracker.get('count', 0) + 1
    
    def get_rate_limit_delay(self) -> float:
        """Get recommended delay based on rate limiting history."""
        if 'last_rate_limit' not in self.rate_limit_tracker:
            return 1.0
        
        last_limit = self.rate_limit_tracker['last_rate_limit']
        count = self.rate_limit_tracker.get('count', 1)
        
        # Increase delay based on frequency of rate limits
        base_delay = 60.0  # 1 minute base delay for rate limits
        multiplier = min(count, 5)  # Cap at 5x multiplier
        
        return base_delay * multiplier
    
    def create_user_notification(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """
        Create a user-friendly notification from error information.
        
        Args:
            error_info: Structured error information
            
        Returns:
            Dictionary with notification data for UI display
        """
        # Map severity to UI notification types
        notification_type_map = {
            ErrorSeverity.INFO: "info",
            ErrorSeverity.WARNING: "warning", 
            ErrorSeverity.ERROR: "error",
            ErrorSeverity.CRITICAL: "error"
        }
        
        notification = {
            'type': notification_type_map[error_info.severity],
            'title': self._get_error_title(error_info),
            'message': error_info.user_message,
            'timestamp': error_info.timestamp.isoformat(),
            'dismissible': error_info.severity in [ErrorSeverity.INFO, ErrorSeverity.WARNING],
            'retry_possible': error_info.retry_possible
        }
        
        if error_info.suggested_action:
            notification['action'] = error_info.suggested_action
        
        if error_info.technical_details and error_info.severity == ErrorSeverity.CRITICAL:
            notification['technical_details'] = error_info.technical_details
        
        return notification
    
    def _get_error_title(self, error_info: ErrorInfo) -> str:
        """Get appropriate title for error notification."""
        title_map = {
            ErrorCategory.API_ERROR: "AI Service Error",
            ErrorCategory.DATA_ERROR: "Data Error", 
            ErrorCategory.VALIDATION_ERROR: "Input Validation Error",
            ErrorCategory.NETWORK_ERROR: "Connection Error",
            ErrorCategory.SYSTEM_ERROR: "System Error",
            ErrorCategory.USER_ERROR: "Input Error"
        }
        
        return title_map.get(error_info.category, "Error")
    
    def log_error(self, error_info: ErrorInfo, context: str = ""):
        """
        Log error information for monitoring and debugging.
        
        Args:
            error_info: Structured error information
            context: Additional context
        """
        # Add to error history
        self.error_history.append(error_info)
        
        # Keep only recent errors (last 100)
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        # Log based on severity
        log_message = f"{context}: {error_info.message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.
        
        Returns:
            Dictionary with error statistics
        """
        if not self.error_history:
            return {'total_errors': 0}
        
        # Count by category and severity
        category_counts = {}
        severity_counts = {}
        
        recent_errors = [
            err for err in self.error_history 
            if err.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors_24h': len(recent_errors),
            'category_breakdown': category_counts,
            'severity_breakdown': severity_counts,
            'rate_limit_info': self.rate_limit_tracker
        }


# Global error handler instance
error_handler = ErrorHandler()