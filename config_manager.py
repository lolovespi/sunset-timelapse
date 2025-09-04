"""
Configuration Manager
Handles loading configuration from YAML files and environment variables
Implements secure practices for sensitive data
"""

import os
import yaml
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigManager:
    """Manages configuration loading and secret handling"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = Path(config_path)
        self.config = {}
        self._load_environment()
        self._load_config()
        self._validate_config()
        
    def _load_environment(self):
        """Load environment variables from .env file if it exists"""
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                
        except Exception as e:
            raise Exception(f"Failed to load configuration: {e}")
            
    def _validate_config(self):
        """Validate required configuration sections exist"""
        required_sections = ['location', 'camera', 'capture', 'video', 'storage', 'youtube']
        missing_sections = []
        
        for section in required_sections:
            if section not in self.config:
                missing_sections.append(section)
                
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
            
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'camera.ip')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
            
    def get_secret(self, env_var_name: str, config_fallback: str = None) -> Optional[str]:
        """
        Get secret from environment variable, with optional config fallback
        
        Args:
            env_var_name: Environment variable name
            config_fallback: Fallback config path (not recommended for secrets)
            
        Returns:
            Secret value or None
        """
        # First try environment variable
        secret = os.getenv(env_var_name)
        
        if secret:
            return secret
            
        # Fallback to config (not recommended for production)
        if config_fallback:
            return self.get(config_fallback)
            
        return None
        
    def get_camera_credentials(self) -> tuple:
        """
        Get camera credentials securely
        
        Returns:
            Tuple of (username, password)
        """
        # Check for username in environment first, then config
        username = self.get_secret('CAMERA_USERNAME') or self.get('camera.username')
        password = self.get_secret('CAMERA_PASSWORD')
        
        if not username or not password:
            raise ValueError("Camera credentials not properly configured. "
                           "Set CAMERA_USERNAME and CAMERA_PASSWORD environment variables.")
            
        return username, password
        
    def get_storage_paths(self) -> Dict[str, Path]:
        """
        Get all storage paths as Path objects
        
        Returns:
            Dictionary of path names to Path objects
        """
        base_path = Path(self.get('storage.base_path'))
        
        paths = {
            'base': base_path,
            'images': base_path / self.get('storage.images_subdir'),
            'videos': base_path / self.get('storage.videos_subdir'),
            'logs': base_path / self.get('storage.logs_subdir'),
            'temp': base_path / self.get('storage.temp_subdir')
        }
        
        # Create directories if they don't exist
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
        return paths
        
    def get_google_credentials_path(self) -> Optional[str]:
        """
        Get Google service account credentials path
        
        Returns:
            Path to credentials file or None
        """
        return self.get_secret('GOOGLE_APPLICATION_CREDENTIALS')
        
    def setup_logging(self) -> None:
        """Setup logging based on configuration"""
        import colorlog
        
        log_level = getattr(logging, self.get('logging.level', 'INFO').upper())
        
        # Create formatter
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Console handler
        if self.get('logging.console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            
        # File handler
        log_paths = self.get_storage_paths()
        log_file = log_paths['logs'] / 'sunset_timelapse.log'
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.get('logging.max_file_size_mb', 10) * 1024 * 1024,
            backupCount=self.get('logging.backup_count', 5)
        )
        
        # Use a simpler formatter for file logs (no colors)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
    def __repr__(self) -> str:
        """String representation of config (without secrets)"""
        safe_config = self.config.copy()
        
        # Remove any potential secrets from representation
        if 'camera' in safe_config and 'password' in safe_config['camera']:
            safe_config['camera']['password'] = '[REDACTED]'
            
        return f"ConfigManager({safe_config})"


# Global config instance
config = None

def get_config(config_path: str = "config.yaml") -> ConfigManager:
    """
    Get global configuration instance
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    global config
    if config is None:
        config = ConfigManager(config_path)
    return config