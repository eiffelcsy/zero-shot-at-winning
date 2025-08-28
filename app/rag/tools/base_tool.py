from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.
    
    Provides a common interface that all tools must implement to ensure
    consistency across the system.
    """
    
    def __init__(self, name: str):
        """
        Initialize the base tool.
        
        Args:
            name: The name identifier for this tool
        """
        self.name = name
    
    @abstractmethod
    async def run(self, query: str, **kwargs) -> Any:
        """
        Execute the tool's main functionality.
        
        Args:
            query: The input query or request
            **kwargs: Additional tool-specific parameters
            
        Returns:
            Tool-specific result
        """
        pass
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate tool inputs before execution.
        
        Args:
            **kwargs: Input parameters to validate
            
        Returns:
            True if inputs are valid, False otherwise
        """
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this tool.
        
        Returns:
            Dictionary containing tool metadata
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__
        }
