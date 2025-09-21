"""
CLI Agent Implementation

This module provides integration with command-line AI agents like Claude Code,
Gemini CLI, and other CLI-based AI tools.
"""

import asyncio
import json
import logging
import subprocess
import os
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

from .cli_config import CLIConfig, AgentConfig
from .utils import timing_decorator, create_error_response


logger = logging.getLogger(__name__)


class CLIAgent:
    """CLI Agent for processing data through command-line AI tools."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize CLI Agent.

        Args:
            config_path: Path to configuration file
        """
        if config_path:
            self.config = CLIConfig.from_file(config_path)
        else:
            self.config = CLIConfig()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            logger.warning(f"Configuration validation errors: {errors}")

        self.current_agent = self.config.default_agent

    @timing_decorator
    async def analyze(self, data: Dict[str, Any], agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze data using specified CLI agent.

        Args:
            data: JSON data to analyze
            agent_name: Name of the agent to use (default: config.default_agent)

        Returns:
            Analysis result
        """
        agent_name = agent_name or self.current_agent
        agent_config = self.config.get_agent(agent_name)

        if not agent_config:
            return {
                "success": False,
                "error": f"Agent '{agent_name}' not found",
                "available_agents": list(self.config.agents.keys())
            }

        if not agent_config.enabled:
            if agent_name == "claude":
                return {
                    "success": False,
                    "error": f"Claude CLI is not installed or configured",
                    "enabled_agents": [a.name for a in self.config.get_enabled_agents()],
                    "installation_hint": "Install Claude CLI from https://docs.anthropic.com/claude/docs/claude-cli and run 'claude auth login' to authenticate"
                }
            else:
                return {
                    "success": False,
                    "error": f"Agent '{agent_name}' is disabled",
                    "enabled_agents": [a.name for a in self.config.get_enabled_agents()]
                }

        try:
            # For Claude Code, we need to inject the JSON data into the prompt
            if agent_name == "claude":
                # Create the prompt with JSON data
                json_str = json.dumps(data, indent=2)
                prompt = f"Use the danny-trades subagent to process this data: {json_str}"

                # Log the prompt for debugging
                logger.info(f"=== CLAUDE PROMPT DEBUG ===")
                logger.info(f"Input data: {data}")
                logger.info(f"Generated JSON: {json_str}")
                logger.info(f"Final prompt: {prompt}")
                logger.info(f"=== END CLAUDE PROMPT DEBUG ===")

                # Prepare command and arguments
                cmd = [agent_config.command] + agent_config.args

                # Check if Claude CLI is available
                try:
                    import shutil
                    shutil.which(cmd[0])
                    if not shutil.which(cmd[0]):
                        return {
                            "success": False,
                            "error": f"Claude CLI not found. Please install Claude CLI first.",
                            "installation_hint": "Visit https://docs.anthropic.com/claude/docs/claude-cli for installation instructions"
                        }
                except Exception:
                    pass

                # Prepare environment variables
                env = os.environ.copy()
                env.update(agent_config.env_vars)

                # Set working directory
                cwd = agent_config.working_dir or os.getcwd()

                # Execute command with stdin input using subprocess in thread executor
                full_cmd = cmd + ['-p', prompt, '--verbose']
                logger.info(f"Executing CLI agent '{agent_name}': {' '.join(full_cmd)}")
                logger.info(f"Prompt length: {len(prompt)} characters")
                logger.debug(f"Prompt content: {prompt[:500]}...")  # Log first 500 chars
                logger.info(f"Full command: {' '.join(full_cmd)}")

                # Execute Claude command
                try:
                    # Escape newlines and special characters for Windows command line
                    import re
                    escaped_prompt = re.sub(r'[\n\r\t"]', ' ', prompt).strip()
                    escaped_prompt = escaped_prompt.replace('\\', '\\\\')

                    logger.info(f"About to execute: {cmd + ['-p', escaped_prompt[:100] + '...', '--verbose']}")

                    # Use subprocess.run in thread executor due to Python 3.13 Windows asyncio limitations
                    def run_claude_command():
                        import subprocess
                        full_command = f'{" ".join(cmd)} -p "{escaped_prompt}" --verbose'
                        logger.info(f"Executing in subprocess: {full_command}")
                        result = subprocess.run(
                            full_command,
                            cwd=cwd,
                            env=env,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=agent_config.timeout
                        )
                        return result.returncode, result.stdout, result.stderr

                    return_code, stdout, stderr = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, run_claude_command),
                        timeout=agent_config.timeout + 10
                    )
                except asyncio.TimeoutError:
                    if 'process' in locals():
                        process.kill()
                        await process.wait()
                    return {
                        "success": False,
                        "error": f"Agent '{agent_name}' timed out after {agent_config.timeout} seconds"
                    }
                except Exception as e:
                    import traceback
                    logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return {
                        "success": False,
                        "error": f"Error executing agent '{agent_name}': {str(e)}",
                        "exception_type": type(e).__name__,
                        "traceback": traceback.format_exc()
                    }

                logger.info(f"Claude process completed with return code: {return_code}")
                logger.info(f"Stdout length: {len(stdout)}")
                logger.info(f"Stderr length: {len(stderr)}")

                # Log actual command output for debugging
                if stdout:
                    logger.info(f"=== CLAUDE STDOUT ===")
                    logger.info(stdout)
                    logger.info(f"=== END CLAUDE STDOUT ===")

                if stderr:
                    logger.warning(f"=== CLAUDE STDERR ===")
                    logger.warning(stderr)
                    logger.info(f"=== END CLAUDE STDERR ===")
            else:
                # For other agents, use the original approach
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(data, f, indent=2)
                    input_file = f.name

                cmd = [agent_config.command] + agent_config.args
                cmd.extend(['--input', input_file])

                # Prepare environment variables
                env = os.environ.copy()
                env.update(agent_config.env_vars)

                # Set working directory
                cwd = agent_config.working_dir or os.getcwd()

                # Execute command
                logger.info(f"Executing CLI agent '{agent_name}': {' '.join(cmd)}")

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=cwd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=agent_config.timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return {
                        "success": False,
                        "error": f"Agent '{agent_name}' timed out after {agent_config.timeout} seconds"
                    }

            # Parse output
            if agent_name == "claude":
                # Handle Claude agent output
                if return_code == 0:
                    try:
                        # Try to parse as JSON first
                        result = json.loads(stdout)
                        return {
                            "success": True,
                            "result": result,
                            "agent": agent_name,
                            "stdout": stdout,
                            "stderr": stderr
                        }
                    except json.JSONDecodeError:
                        # Return raw output if not JSON
                        return {
                            "success": True,
                            "result": stdout,
                            "agent": agent_name,
                            "stdout": stdout,
                            "stderr": stderr
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Agent '{agent_name}' failed with return code {return_code}",
                        "stdout": stdout,
                        "stderr": stderr
                    }
            else:
                # Handle other agents
                # Clean up temporary files
                try:
                    os.unlink(input_file)
                except OSError:
                    pass

                if return_code == 0:
                    try:
                        # Try to parse as JSON first
                        result = json.loads(stdout)
                        return {
                            "success": True,
                            "result": result,
                            "agent": agent_name,
                            "stdout": stdout,
                            "stderr": stderr
                        }
                    except json.JSONDecodeError:
                        # Return raw output if not JSON
                        return {
                            "success": True,
                            "result": stdout,
                            "agent": agent_name,
                            "stdout": stdout,
                            "stderr": stderr
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Agent '{agent_name}' failed with return code {return_code}",
                        "stdout": stdout,
                        "stderr": stderr
                    }

        except Exception as e:
            import traceback
            logger.error(f"Error executing agent '{agent_name}': {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Error executing agent '{agent_name}': {str(e)}",
                "traceback": traceback.format_exc()
            }

    async def analyze_with_prompt(self, data: Dict[str, Any], prompt: str, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze data using specified CLI agent with custom prompt.

        Args:
            data: JSON data to analyze
            prompt: Custom prompt to use
            agent_name: Name of the agent to use (default: config.default_agent)

        Returns:
            Analysis result
        """
        agent_name = agent_name or self.current_agent
        agent_config = self.config.get_agent(agent_name)

        if not agent_config:
            return {
                "success": False,
                "error": f"Agent '{agent_name}' not found",
                "available_agents": list(self.config.agents.keys())
            }

        try:
            # For Claude Code, combine data and prompt into stdin
            if agent_name == "claude":
                json_str = json.dumps(data, indent=2)
                full_prompt = f"{prompt}\n\nUse the danny-trades subagent to process this data: {json_str}"

                # Prepare command and arguments
                cmd = [agent_config.command] + agent_config.args

                # Prepare environment variables
                env = os.environ.copy()
                env.update(agent_config.env_vars)

                # Set working directory
                cwd = agent_config.working_dir or os.getcwd()

                # Execute command with stdin input using subprocess in thread executor
                full_cmd = cmd + ['-p', full_prompt, '--verbose']
                logger.info(f"Executing CLI agent '{agent_name}' with custom prompt: {' '.join(full_cmd)}")
                logger.info(f"Full prompt length: {len(full_prompt)} characters")
                logger.debug(f"Full prompt content: {full_prompt[:500]}...")  # Log first 500 chars
                logger.info(f"Full command: {' '.join(full_cmd)}")

                # Execute Claude command
                try:
                    # Escape newlines and special characters for Windows command line
                    import re
                    escaped_full_prompt = re.sub(r'[\n\r\t"]', ' ', full_prompt).strip()
                    escaped_full_prompt = escaped_full_prompt.replace('\\', '\\\\')

                    logger.info(f"About to execute: {cmd + ['-p', escaped_full_prompt[:100] + '...', '--verbose']}")

                    # Use subprocess.run in thread executor due to Python 3.13 Windows asyncio limitations
                    def run_claude_command():
                        import subprocess
                        full_command = f'{" ".join(cmd)} -p "{escaped_full_prompt}" --verbose'
                        logger.info(f"Executing in subprocess: {full_command}")
                        result = subprocess.run(
                            full_command,
                            cwd=cwd,
                            env=env,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=agent_config.timeout
                        )
                        return result.returncode, result.stdout, result.stderr

                    return_code, stdout, stderr = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, run_claude_command),
                        timeout=agent_config.timeout + 10
                    )
                except asyncio.TimeoutError:
                    if 'process' in locals():
                        process.kill()
                        await process.wait()
                    return {
                        "success": False,
                        "error": f"Agent '{agent_name}' timed out after {agent_config.timeout} seconds"
                    }
                except Exception as e:
                    import traceback
                    logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return {
                        "success": False,
                        "error": f"Error executing agent '{agent_name}': {str(e)}",
                        "exception_type": type(e).__name__,
                        "traceback": traceback.format_exc()
                    }

                logger.info(f"Claude process completed with return code: {return_code}")
                logger.info(f"Stdout length: {len(stdout)}")
                logger.info(f"Stderr length: {len(stderr)}")

                # Log actual command output for debugging
                if stdout:
                    logger.info(f"=== CLAUDE STDOUT ===")
                    logger.info(stdout)
                    logger.info(f"=== END CLAUDE STDOUT ===")

                if stderr:
                    logger.warning(f"=== CLAUDE STDERR ===")
                    logger.warning(stderr)
                    logger.info(f"=== END CLAUDE STDERR ===")
            else:
                # For other agents, use the original approach
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(data, f, indent=2)
                    input_file = f.name

                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(prompt)
                    prompt_file = f.name

                # Prepare command and arguments
                cmd = [agent_config.command] + agent_config.args

                # Add input and prompt files to command
                cmd.extend(['--input', input_file, '--prompt', prompt_file])

                # Prepare environment variables
                env = os.environ.copy()
                env.update(agent_config.env_vars)

                # Set working directory
                cwd = agent_config.working_dir or os.getcwd()

                # Execute command
                logger.info(f"Executing CLI agent '{agent_name}' with custom prompt: {' '.join(cmd)}")

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=cwd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=agent_config.timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return {
                        "success": False,
                        "error": f"Agent '{agent_name}' timed out after {agent_config.timeout} seconds"
                    }

            # Parse output
            if agent_name == "claude":
                # Handle Claude agent output
                if return_code == 0:
                    try:
                        result = json.loads(stdout)
                        return {
                            "success": True,
                            "result": result,
                            "agent": agent_name,
                            "stdout": stdout,
                            "stderr": stderr
                        }
                    except json.JSONDecodeError:
                        return {
                            "success": True,
                            "result": stdout,
                            "agent": agent_name,
                            "stdout": stdout,
                            "stderr": stderr
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Agent '{agent_name}' failed with return code {return_code}",
                        "stdout": stdout,
                        "stderr": stderr
                    }
            else:
                # Handle other agents
                # Clean up temporary files
                for temp_file in [input_file, prompt_file]:
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass

                if return_code == 0:
                    try:
                        result = json.loads(stdout)
                        return {
                            "success": True,
                            "result": result,
                            "agent": agent_name,
                            "stdout": stdout,
                            "stderr": stderr
                        }
                    except json.JSONDecodeError:
                        return {
                            "success": True,
                            "result": stdout,
                            "agent": agent_name,
                            "stdout": stdout,
                            "stderr": stderr
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Agent '{agent_name}' failed with return code {return_code}",
                        "stdout": stdout,
                        "stderr": stderr
                    }

        except Exception as e:
            import traceback
            logger.error(f"Error executing agent '{agent_name}': {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Error executing agent '{agent_name}': {str(e)}",
                "traceback": traceback.format_exc()
            }

    def get_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return list(self.config.agents.keys())

    def get_enabled_agents(self) -> List[str]:
        """Get list of enabled agent names."""
        return [agent.name for agent in self.config.get_enabled_agents()]

    def set_current_agent(self, agent_name: str) -> bool:
        """Set the current agent to use."""
        if agent_name in self.config.agents:
            self.current_agent = agent_name
            return True
        return False

    def get_agent_status(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status information for an agent."""
        agent_name = agent_name or self.current_agent
        agent_config = self.config.get_agent(agent_name)

        if not agent_config:
            return {
                "name": agent_name,
                "status": "not_found",
                "error": f"Agent '{agent_name}' not configured"
            }

        return {
            "name": agent_name,
            "status": "enabled" if agent_config.enabled else "disabled",
            "command": agent_config.command,
            "args": agent_config.args,
            "timeout": agent_config.timeout,
            "working_dir": agent_config.working_dir,
            "env_vars": {k: "***" if "key" in k.lower() else v for k, v in agent_config.env_vars.items()}
        }

    async def test_agent(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Test if an agent is available and working."""
        agent_name = agent_name or self.current_agent
        agent_config = self.config.get_agent(agent_name)

        if not agent_config:
            return {
                "success": False,
                "error": f"Agent '{agent_name}' not found"
            }

        try:
            # Test with a simple command
            cmd = [agent_config.command] + agent_config.args + ["--help"]

            env = os.environ.copy()
            env.update(agent_config.env_vars)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30
            )

            return {
                "success": process.returncode == 0,
                "agent": agent_name,
                "return_code": process.returncode,
                "stdout": stdout[:500],  # Limit output
                "stderr": stderr[:500]
            }

        except Exception as e:
            return {
                "success": False,
                "agent": agent_name,
                "error": str(e)
            }