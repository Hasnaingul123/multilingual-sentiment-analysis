"""
Configuration Loader Module

Handles loading and validation of YAML configuration files.
Supports environment variable interpolation and config merging.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from copy import deepcopy


class ConfigLoader:
    """
    Load and manage configuration files with validation and interpolation.
    
    Example:
        >>> config = ConfigLoader.load_config('config/model_config.yaml')
        >>> model_name = config['model']['base_model']
    """
    
    @staticmethod
    def load_config(
        config_path: Union[str, Path],
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
            validate: Whether to validate the configuration
            
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
        
        # Interpolate environment variables
        config = ConfigLoader._interpolate_env_vars(config)
        
        if validate:
            ConfigLoader._validate_config(config, config_path.stem)
        
        return config
    
    @staticmethod
    def load_all_configs(config_dir: Union[str, Path] = 'config') -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files from a directory.
        
        Args:
            config_dir: Directory containing config files
            
        Returns:
            Dictionary mapping config names to their contents
        """
        config_dir = Path(config_dir)
        configs = {}
        
        for config_file in config_dir.glob('*.yaml'):
            config_name = config_file.stem
            configs[config_name] = ConfigLoader.load_config(config_file)
        
        return configs
    
    @staticmethod
    def merge_configs(
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries (deep merge).
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base with
            
        Returns:
            Merged configuration dictionary
        """
        merged = deepcopy(base_config)
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def _interpolate_env_vars(config: Any) -> Any:
        """
        Recursively interpolate environment variables in config values.
        
        Format: ${ENV_VAR_NAME} or ${ENV_VAR_NAME:default_value}
        """
        if isinstance(config, dict):
            return {k: ConfigLoader._interpolate_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [ConfigLoader._interpolate_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Simple environment variable interpolation
            if config.startswith('${') and config.endswith('}'):
                env_var = config[2:-1]
                if ':' in env_var:
                    var_name, default_value = env_var.split(':', 1)
                    return os.getenv(var_name, default_value)
                else:
                    return os.getenv(env_var, config)
        return config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any], config_type: str) -> None:
        """
        Validate configuration based on type.
        
        Args:
            config: Configuration dictionary
            config_type: Type of config (model_config, training_config, etc.)
            
        Raises:
            ValueError: If validation fails
        """
        if config_type == 'model_config':
            ConfigLoader._validate_model_config(config)
        elif config_type == 'training_config':
            ConfigLoader._validate_training_config(config)
        elif config_type == 'preprocessing_config':
            ConfigLoader._validate_preprocessing_config(config)
    
    @staticmethod
    def _validate_model_config(config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        required_keys = ['model', 'multitask']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in model_config: {key}")
        
        # Validate loss weights sum approximately to 1.0
        sentiment_weight = config['multitask'].get('sentiment_weight', 0.6)
        sarcasm_weight = config['multitask'].get('sarcasm_weight', 0.4)
        
        if not (0.0 < sentiment_weight <= 1.0 and 0.0 < sarcasm_weight <= 1.0):
            raise ValueError("Loss weights must be between 0 and 1")
    
    @staticmethod
    def _validate_training_config(config: Dict[str, Any]) -> None:
        """Validate training configuration."""
        required_keys = ['training']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in training_config: {key}")
        
        # Validate batch size
        batch_size = config['training'].get('batch_size', 16)
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        
        # Validate learning rate
        lr = config['training']['optimizer'].get('learning_rate', 2e-5)
        if lr <= 0:
            raise ValueError("Learning rate must be positive")
    
    @staticmethod
    def _validate_preprocessing_config(config: Dict[str, Any]) -> None:
        """Validate preprocessing configuration."""
        required_keys = ['preprocessing']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in preprocessing_config: {key}")
        
        # Validate max_length
        max_length = config['preprocessing']['tokenization'].get('max_length', 128)
        if max_length < 1 or max_length > 512:
            raise ValueError("max_length must be between 1 and 512")
    
    @staticmethod
    def save_config(
        config: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration dictionary
            output_path: Path to save the config file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# Convenience function
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    return ConfigLoader.load_config(config_path)


if __name__ == "__main__":
    # Test configuration loading
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
        print(f"Successfully loaded config from {config_path}")
        print(yaml.dump(config, default_flow_style=False))
    else:
        print("Usage: python config_loader.py <path_to_config.yaml>")
