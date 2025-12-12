from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import logging
from llm.llm_client import Message

@dataclass
class AgentOutput:
    success: bool
    data: Any
    message: str
    logs: List[str] = field(default_factory=list)

class BaseAgent(ABC):
    def __init__(self, name: str, llm_client, config):
        self.name = name
        self.llm_client = llm_client
        self.config = config
        self.tools = {}
        self.logger = logging.getLogger(name)
        self._register_tools()
        
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass
    
    @abstractmethod
    def _register_tools(self):
        pass
    
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        pass
    
    def register_tool(
        self,
        name: str,
        func: callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        self.tools[name] = {
            "function": func,
            "definition": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self.logger.info(f"Registered tool: {name}")
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:

        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        self.logger.info(f"Executing tool: {name} with args: {arguments}")

        return self.tools[name]["function"](**arguments)
    
    def get_tool_definitions(self) -> List[Dict]:

        return [tool["definition"] for tool in self.tools.values()]

    def think(self, prompt: str, context: Optional[Dict] = None) -> str:

        messages = [
            Message(role="system", content=self.system_prompt)
        ]

        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            messages.append(Message(role="user", content=f"Context:\n{context_str}"))

        messages.append(Message(role="user", content=prompt))

        self.logger.debug(f"Thinking about: {prompt[:100]}...")
        response = self.llm_client.chat(messages)

        return response.content

    def think_with_tools(self, prompt: str, context: Optional[Dict] = None) -> AgentOutput:

        logs = []

        messages = [
            Message(role="system", content=self.system_prompt)
        ]

        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            messages.append(Message(role="user", content=f"Context:\n{context_str}"))
            logs.append("Added context to prompt")

        messages.append(Message(role="user", content=prompt))

        tools = self.get_tool_definitions()

        if not tools:
            response = self.llm_client.chat(messages)
            return AgentOutput(
                success=True,
                data={"response": response.content},
                message="Completed without tools",
                logs=logs
            )

        self.logger.debug(f"Thinking with {len(tools)} tools about: {prompt[:100]}...")
        logs.append(f"Starting ReAct loop with {len(tools)} tools")

        try:
            response = self.llm_client.chat_with_tools(
                messages=messages,
                tools=tools,
                tool_executor=self.execute_tool
            )

            logs.append("ReAct loop completed")

            return AgentOutput(
                success=True,
                data={"response": response.content},
                message="Completed with tools",
                logs=logs
            )

        except Exception as e:
            self.logger.error(f"Error in think_with_tools: {e}")
            return AgentOutput(
                success=False,
                data={},
                message=f"Error: {str(e)}",
                logs=logs
            )
