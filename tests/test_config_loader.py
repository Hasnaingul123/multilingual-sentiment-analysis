"""
Test suite for configuration loader
"""

import pytest
import yaml
from pathlib import Path
import tempfile
import shutil

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config_loader import ConfigLoader, load_config


class TestConfigLoader:
    """Test cases for ConfigLoader class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for test configs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration dictionary."""
        return {
            'model': {
                'base_model': 'xlm-roberta-base',
                'hidden_size': 768
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 2e-5
            }
        }
    
    def test_load_valid_config(self, temp_config_dir, sample_config):
        """Test loading a valid configuration file."""
        config_file = temp_config_dir / 'test_config.yaml'
        
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        loaded_config = load_config(config_file)
        
        assert loaded_config == sample_config
        assert loaded_config['model']['base_model'] == 'xlm-roberta-base'
        assert loaded_config['training']['batch_size'] == 16
    
    def test_load_nonexistent_config(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.yaml')
    
    def test_merge_configs(self, sample_config):
        """Test merging two configuration dictionaries."""
        override_config = {
            'model': {
                'hidden_size': 1024  # Override
            },
            'training': {
                'num_epochs': 20  # Add new key
            }
        }
        
        merged = ConfigLoader.merge_configs(sample_config, override_config)
        
        assert merged['model']['base_model'] == 'xlm-roberta-base'  # Preserved
        assert merged['model']['hidden_size'] == 1024  # Overridden
        assert merged['training']['num_epochs'] == 20  # Added
        assert merged['training']['batch_size'] == 16  # Preserved
    
    def test_env_var_interpolation(self, temp_config_dir, monkeypatch):
        """Test environment variable interpolation."""
        monkeypatch.setenv('TEST_MODEL_NAME', 'bert-base-multilingual')
        
        config_with_env = {
            'model': {
                'base_model': '${TEST_MODEL_NAME}',
                'fallback': '${NONEXISTENT:default_value}'
            }
        }
        
        config_file = temp_config_dir / 'env_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_with_env, f)
        
        loaded_config = load_config(config_file)
        
        assert loaded_config['model']['base_model'] == 'bert-base-multilingual'
        assert loaded_config['model']['fallback'] == 'default_value'
    
    def test_save_config(self, temp_config_dir, sample_config):
        """Test saving configuration to file."""
        output_file = temp_config_dir / 'output_config.yaml'
        
        ConfigLoader.save_config(sample_config, output_file)
        
        assert output_file.exists()
        
        # Load and verify
        loaded_config = load_config(output_file)
        assert loaded_config == sample_config


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
