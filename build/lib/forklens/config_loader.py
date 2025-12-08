import yaml
import os

class Config:
    """
    Configuration loader class.
    
    Loads a YAML configuration file and creates a nested object structure
    for easy access to configuration parameters.
    """
    def __init__(self, config_path):
        """
        Initialize the Config loader.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self._config = self._load_config()
        self._create_attributes(self._config)

    def _load_config(self):
        """
        Load the YAML configuration file.
        
        Returns:
            dict: The loaded configuration dictionary.
            
        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_attributes(self, config_dict):
        """
        Recursively create attributes from the configuration dictionary.
        
        Args:
            config_dict (dict): The configuration dictionary.
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

class ConfigNode:
    """
    A node in the configuration tree.
    
    Allows accessing configuration values as attributes or dictionary items.
    """
    def __init__(self, config_dict):
        """
        Initialize a ConfigNode.
        
        Args:
            config_dict (dict): A dictionary representing a section of the configuration.
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, item):
        """
        Allow dictionary-style access to configuration values.
        
        Args:
            item (str): The key to access.
            
        Returns:
            The value associated with the key.
        """
        return getattr(self, item)
