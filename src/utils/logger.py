"""
Logging Module

Provides structured logging with file and console output.
Supports different log levels, formatting, and log rotation.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class Logger:
    """
    Custom logger with file and console handlers.
    
    Features:
    - Dual output (console + file)
    - Structured formatting
    - Log rotation
    - Configurable log levels
    
    Example:
        >>> logger = Logger.get_logger('model_training')
        >>> logger.info("Starting training...")
        >>> logger.error("Training failed", exc_info=True)
    """
    
    _loggers = {}  # Cache for loggers
    
    @staticmethod
    def get_logger(
        name: str,
        log_dir: Union[str, Path] = 'logs',
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True,
        rotation: str = 'size'  # 'size' or 'time'
    ) -> logging.Logger:
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name (typically module name)
            log_dir: Directory to store log files
            log_file: Log file name (auto-generated if None)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Enable console output
            file_output: Enable file output
            rotation: Rotation strategy ('size' or 'time')
            
        Returns:
            Configured logger instance
        """
        # Return cached logger if exists
        if name in Logger._loggers:
            return Logger._loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers = []  # Clear existing handlers
        
        # Create formatter
        formatter = Logger._create_formatter()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            if log_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = f"{name}_{timestamp}.log"
            
            log_path = log_dir / log_file
            
            if rotation == 'size':
                # Rotate when file reaches 10MB, keep 5 backups
                file_handler = RotatingFileHandler(
                    log_path,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                )
            else:  # time-based rotation
                # Rotate daily, keep 30 days
                file_handler = TimedRotatingFileHandler(
                    log_path,
                    when='midnight',
                    interval=1,
                    backupCount=30,
                    encoding='utf-8'
                )
            
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        # Cache logger
        Logger._loggers[name] = logger
        
        return logger
    
    @staticmethod
    def _create_formatter() -> logging.Formatter:
        """
        Create a structured log formatter.
        
        Returns:
            Configured formatter
        """
        log_format = (
            '%(asctime)s | %(levelname)-8s | %(name)s | '
            '%(filename)s:%(lineno)d | %(message)s'
        )
        
        date_format = '%Y-%m-%d %H:%M:%S'
        
        return logging.Formatter(log_format, datefmt=date_format)
    
    @staticmethod
    def setup_default_logger(
        log_dir: Union[str, Path] = 'logs',
        level: int = logging.INFO
    ) -> logging.Logger:
        """
        Setup a default application logger.
        
        Args:
            log_dir: Directory for log files
            level: Logging level
            
        Returns:
            Default logger instance
        """
        return Logger.get_logger(
            name='multilingual_sentiment',
            log_dir=log_dir,
            level=level,
            console_output=True,
            file_output=True
        )
    
    @staticmethod
    def set_level(logger: logging.Logger, level: int) -> None:
        """
        Change the logging level for a logger and all its handlers.
        
        Args:
            logger: Logger instance
            level: New logging level
        """
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


class MetricsLogger:
    """
    Logger for tracking training metrics.
    
    Example:
        >>> metrics_logger = MetricsLogger('logs/metrics.log')
        >>> metrics_logger.log_metrics(epoch=1, loss=0.5, accuracy=0.85)
    """
    
    def __init__(self, log_file: Union[str, Path]):
        """
        Initialize metrics logger.
        
        Args:
            log_file: Path to metrics log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create CSV header if file doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                f.write("timestamp,epoch,step,metric_name,metric_value\n")
    
    def log_metrics(self, **metrics) -> None:
        """
        Log metrics to file.
        
        Args:
            **metrics: Keyword arguments of metric_name=value
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a') as f:
            for metric_name, metric_value in metrics.items():
                # Handle nested dictionaries
                if isinstance(metric_value, dict):
                    for sub_name, sub_value in metric_value.items():
                        full_name = f"{metric_name}_{sub_name}"
                        f.write(f"{timestamp},{full_name},{sub_value}\n")
                else:
                    f.write(f"{timestamp},{metric_name},{metric_value}\n")
    
    def log_epoch_metrics(
        self,
        epoch: int,
        step: int,
        **metrics
    ) -> None:
        """
        Log metrics for a specific epoch and step.
        
        Args:
            epoch: Current epoch number
            step: Current step number
            **metrics: Metric values
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a') as f:
            for metric_name, metric_value in metrics.items():
                f.write(f"{timestamp},{epoch},{step},{metric_name},{metric_value}\n")


# Convenience functions
def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for Logger.get_logger
        
    Returns:
        Logger instance
    """
    return Logger.get_logger(name, **kwargs)


def setup_logging(log_dir: str = 'logs', level: int = logging.INFO) -> logging.Logger:
    """
    Setup default logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Default logger
    """
    return Logger.setup_default_logger(log_dir, level)


if __name__ == "__main__":
    # Test logging functionality
    logger = setup_logging()
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test metrics logger
    metrics_logger = MetricsLogger('logs/test_metrics.log')
    metrics_logger.log_epoch_metrics(
        epoch=1,
        step=100,
        loss=0.5,
        accuracy=0.85,
        f1_score=0.82
    )
    
    print("Logging test completed successfully!")
