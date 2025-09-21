"""
CLI Agent Configuration Module

This module provides configuration management for CLI agents like Claude Code,
Gemini CLI, and other command-line AI tools.
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for a specific CLI agent."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    timeout: int = 300
    enabled: bool = True


@dataclass
class CLIConfig:
    """Main configuration class for CLI agents."""
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    default_agent: str = "claude"
    global_timeout: int = 300
    log_level: str = "DEBUG"
    max_retries: int = 3

    def __post_init__(self):
        """Initialize default agents if none provided."""
        if not self.agents:
            self.agents = self._get_default_agents()

    def _get_default_agents(self) -> Dict[str, AgentConfig]:
        """Get default CLI agent configurations."""
        import shutil

        # Try to find Claude CLI in the system
        claude_path = shutil.which("claude")
        claude_enabled = claude_path is not None

        return {
            "claude": AgentConfig(
                name="claude",
                command=claude_path if claude_path else "claude",
                args=[],
                env_vars={},
                timeout=60,  # Reduced timeout for better responsiveness
                enabled=claude_enabled
            ),
            "gemini": AgentConfig(
                name="gemini",
                command="gemini",
                args=["--cli"],
                env_vars={
                    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
                },
                timeout=300,
                enabled=bool(os.getenv("GEMINI_API_KEY"))
            ),
            "openai": AgentConfig(
                name="openai",
                command="openai",
                args=["--api"],
                env_vars={
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                },
                timeout=300,
                enabled=bool(os.getenv("OPENAI_API_KEY"))
            )
        }

    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name."""
        return self.agents.get(name)

    def get_enabled_agents(self) -> List[AgentConfig]:
        """Get all enabled agents."""
        return [agent for agent in self.agents.values() if agent.enabled]

    def add_agent(self, agent: AgentConfig) -> None:
        """Add a new agent configuration."""
        self.agents[agent.name] = agent

    def remove_agent(self, name: str) -> bool:
        """Remove an agent configuration."""
        if name in self.agents:
            del self.agents[name]
            return True
        return False

    def update_agent(self, name: str, **kwargs) -> bool:
        """Update an existing agent configuration."""
        if name not in self.agents:
            return False

        agent = self.agents[name]
        for key, value in kwargs.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
        return True

    @classmethod
    def from_file(cls, config_path: str) -> "CLIConfig":
        """Load configuration from file."""
        path = Path(config_path)

        if not path.exists():
            return cls()

        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # Convert dict to AgentConfig objects
        agents = {}
        for name, agent_data in data.get('agents', {}).items():
            agents[name] = AgentConfig(**agent_data)

        return cls(
            agents=agents,
            default_agent=data.get('default_agent', 'claude'),
            global_timeout=data.get('global_timeout', 300),
            log_level=data.get('log_level', 'INFO'),
            max_retries=data.get('max_retries', 3)
        )

    def to_file(self, config_path: str) -> None:
        """Save configuration to file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'default_agent': self.default_agent,
            'global_timeout': self.global_timeout,
            'log_level': self.log_level,
            'max_retries': self.max_retries,
            'agents': {
                name: {
                    'name': agent.name,
                    'command': agent.command,
                    'args': agent.args,
                    'env_vars': agent.env_vars,
                    'working_dir': agent.working_dir,
                    'timeout': agent.timeout,
                    'enabled': agent.enabled
                }
                for name, agent in self.agents.items()
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.agents:
            errors.append("No agents configured")

        if self.default_agent not in self.agents:
            errors.append(f"Default agent '{self.default_agent}' not found")

        for name, agent in self.agents.items():
            if not agent.command:
                errors.append(f"Agent '{name}' has no command")

            if agent.timeout <= 0:
                errors.append(f"Agent '{name}' has invalid timeout")

        return errors