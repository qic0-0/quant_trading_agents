from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import json
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
    content: str

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]

@dataclass
class LLMResponse:
    content: str
    tool_calls: Optional[List[ToolCall]] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = field(default_factory=dict)
    raw_response: Optional[Dict] = field(default_factory=dict)


class LLMClient:
    def __init__(self, config):
        self.config = config
        self.api_base_url = config.api_base_url
        self.model_name = config.model_name
        self.max_retries = getattr(config, 'max_retries', 3)
        self.timeout = getattr(config, 'timeout', 60)
        try:
            from openai import OpenAI

            if hasattr(config, 'api_key') and config.api_key:
                api_key = config.api_key
            else:
                try:
                    from inference_auth_token import get_access_token
                    api_key = get_access_token()
                    logger.info("Using Argonne access token")
                except ImportError:
                    logger.warning("inference_auth_token not found, using config api_key")
                    api_key = getattr(config, 'api_key', '')

            self.client = OpenAI(
                api_key=api_key,
                base_url=self.api_base_url
            )
            logger.info(f"LLM client initialized with base URL: {self.api_base_url}")

        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            self.client = None

    def chat(self,messages: List[Message], tools: Optional[List[Dict]] = None, temperature: Optional[float] = None) -> LLMResponse:
        if self.client is None:
            logger.error("LLM client not initialized")
            return LLMResponse(content="Error: LLM client not initialized")

        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        kwargs = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": temperature if temperature is not None else 0.7,
        }

        if tools:
            kwargs["tools"] = self._format_tools(tools)
            kwargs["tool_choice"] = "auto"

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return self._parse_response(response)

            except Exception as e:
                logger.warning(f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"LLM request failed after {self.max_retries} attempts")
                    return LLMResponse(content=f"Error: {str(e)}")

        return LLMResponse(content="Error: Max retries exceeded")

    def chat_with_tools(self, messages: List[Message], tools: List[Dict], tool_executor: Callable) -> LLMResponse:
        max_iterations = 10
        current_messages = list(messages)

        for iteration in range(max_iterations):
            logger.debug(f"Tool loop iteration {iteration + 1}/{max_iterations}")
            response = self.chat(current_messages, tools)
            if not response.tool_calls:
                logger.debug("No tool calls, returning final response")
                return response
            current_messages.append(Message(
                role="assistant",
                content=response.content or ""
            ))
            for tool_call in response.tool_calls:
                logger.info(f"Executing tool: {tool_call.name}")

                try:
                    result = tool_executor(tool_call.name, tool_call.arguments)
                    result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    result_str = f"Error executing tool: {str(e)}"

                current_messages.append(Message(
                    role="user",
                    content=f"Tool '{tool_call.name}' result: {result_str}"
                ))

        logger.warning(f"Max iterations ({max_iterations}) reached in tool loop")
        return response

    def _format_tools(self, tools: List[Dict]) -> List[Dict]:

        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                }
            })
        return formatted

    def _parse_response(self, raw_response) -> LLMResponse:

        try:
            choice = raw_response.choices[0]
            message = choice.message
            content = message.content or ""
            tool_calls = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    tool_calls.append(ToolCall(
                        name=tc.function.name,
                        arguments=arguments
                    ))

            usage = {}
            if hasattr(raw_response, 'usage') and raw_response.usage:
                usage = {
                    "prompt_tokens": raw_response.usage.prompt_tokens,
                    "completion_tokens": raw_response.usage.completion_tokens,
                    "total_tokens": raw_response.usage.total_tokens
                }

            return LLMResponse(
                content=content,
                tool_calls=tool_calls if tool_calls else None,
                usage=usage,
                raw_response=raw_response.model_dump() if hasattr(raw_response, 'model_dump') else {}
            )

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return LLMResponse(content=f"Error parsing response: {str(e)}")
