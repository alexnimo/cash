from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel

class PluginMetadata(BaseModel):
    name: str
    version: str
    description: str
    author: str

class BasePlugin(ABC):
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass

    @abstractmethod
    async def process(self, data: Any) -> Dict[str, Any]:
        """Process the input data and return results"""
        pass

class PluginManager:
    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}

    def register_plugin(self, plugin: BasePlugin):
        metadata = plugin.get_metadata()
        self._plugins[metadata.name] = plugin

    def get_plugin(self, name: str) -> BasePlugin:
        return self._plugins.get(name)

    def list_plugins(self) -> List[PluginMetadata]:
        return [plugin.get_metadata() for plugin in self._plugins.values()]

    async def run_plugin(self, name: str, data: Any) -> Dict[str, Any]:
        plugin = self.get_plugin(name)
        if not plugin:
            raise ValueError(f"Plugin {name} not found")
        return await plugin.process(data)
