"""LiteLLM-based LLM Client for SuperAgent.

Uses litellm module for OpenRouter provider instead of term_sdk.
Compatible with the existing agent loop interface.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import litellm
from litellm import completion


@dataclass
class FunctionCall:
    """Represents a function/tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]
    
    @classmethod
    def from_openai(cls, call: Any) -> "FunctionCall":
        """Parse from OpenAI tool_calls format."""
        func = call.function if hasattr(call, "function") else call.get("function", {})
        
        if hasattr(func, "arguments"):
            args_str = func.arguments
        else:
            args_str = func.get("arguments", "{}")
        
        if hasattr(func, "name"):
            name = func.name
        else:
            name = func.get("name", "")
        
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {"raw": args_str}
        
        call_id = call.id if hasattr(call, "id") else call.get("id", "")
        
        return cls(
            id=call_id,
            name=name,
            arguments=args,
        )


@dataclass
class TokenUsage:
    """Token usage information."""
    input: int = 0
    output: int = 0
    cached: int = 0
    
    @property
    def total(self) -> int:
        return self.input + self.output


@dataclass
class LLMResponse:
    """Response from the LLM."""
    text: str = ""
    function_calls: List[FunctionCall] = field(default_factory=list)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    finish_reason: str = ""
    cost: float = 0.0
    raw: Optional[Any] = None
    
    def has_function_calls(self) -> bool:
        """Check if response contains function calls."""
        return len(self.function_calls) > 0


class LiteLLMClient:
    """LLM Client using litellm for OpenRouter provider."""
    
    def __init__(
        self,
        provider: str = "openrouter",
        default_model: str = "anthropic/claude-3.5-sonnet",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        timeout: int = 300,
    ):
        """Initialize the LiteLLM client.
        
        Args:
            provider: Provider name (openrouter supported)
            default_model: Default model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Stats tracking
        self._total_tokens = 0
        self._total_cost = 0.0
        self._request_count = 0
        
        # Configure litellm
        self._configure_litellm()
    
    def _configure_litellm(self) -> None:
        """Configure litellm settings."""
        # Set API key from environment
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
        
        # Check for LLM_PROXY_URL (used in evaluation mode)
        proxy_url = os.environ.get("LLM_PROXY_URL")
        if proxy_url:
            # In evaluation mode, use the proxy URL as base
            litellm.api_base = proxy_url
            self._log(f"Using LLM proxy: {proxy_url}")
        
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
        
        # Optional: Set site URL and app name for OpenRouter
        site_url = os.environ.get("OR_SITE_URL", "https://term-challenge.platform.network")
        app_name = os.environ.get("OR_APP_NAME", "term-challenge-agent")
        os.environ["OR_SITE_URL"] = site_url
        os.environ["OR_APP_NAME"] = app_name
        
        # Disable litellm logging unless debug mode
        litellm.suppress_debug_info = True
    
    def _get_model_name(self, model: Optional[str] = None) -> str:
        """Get the model name for the API call."""
        model = model or self.default_model
        
        # For OpenRouter, prefix with openrouter/
        if self.provider == "openrouter" and not model.startswith("openrouter/"):
            return f"openrouter/{model}"
        
        return model
    
    def _supports_temperature(self, model: str) -> bool:
        """Check if the model supports the temperature parameter.
        
        Reasoning models like o1, o3, deepseek-r1 don't support temperature.
        """
        model_lower = model.lower()
        # OpenAI reasoning models
        if model_lower.startswith("o1") or model_lower.startswith("o3"):
            return False
        if "/o1" in model_lower or "/o3" in model_lower:
            return False
        # DeepSeek reasoning models
        if "deepseek-r1" in model_lower or "deepseek/deepseek-r1" in model_lower:
            return False
        return True
    
    def _build_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Build tools list for litellm."""
        if not tools:
            return None
        
        result = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return result
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse litellm response to LLMResponse."""
        result = LLMResponse(raw=response)
        
        # Parse usage
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            result.tokens = TokenUsage(
                input=getattr(usage, "prompt_tokens", 0) or 0,
                output=getattr(usage, "completion_tokens", 0) or 0,
                cached=getattr(usage, "prompt_tokens_details", {}).get("cached_tokens", 0) if hasattr(usage, "prompt_tokens_details") else 0,
            )
        
        # Parse model
        result.model = getattr(response, "model", self.default_model) or self.default_model
        
        # Parse choices
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message if hasattr(choice, "message") else None
            
            if message:
                result.finish_reason = getattr(choice, "finish_reason", "") or ""
                result.text = getattr(message, "content", "") or ""
                
                # Parse function calls
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    for call in tool_calls:
                        result.function_calls.append(FunctionCall.from_openai(call))
        
        # Calculate cost using litellm's cost tracking
        try:
            cost = litellm.completion_cost(completion_response=response)
            result.cost = cost or 0.0
        except Exception:
            result.cost = 0.0
        
        return result
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat request to the LLM.
        
        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tools/functions
            model: Model to use (defaults to default_model)
            max_tokens: Max tokens to generate
            temperature: Temperature for generation
            extra_body: Extra parameters to pass to the API
            **kwargs: Additional arguments
            
        Returns:
            LLMResponse with text and/or function calls
        """
        model_name = self._get_model_name(model)
        max_tokens = max_tokens or self.max_tokens
        
        # Build request kwargs
        request_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "timeout": self.timeout,
        }
        
        # Only add temperature if model supports it
        if self._supports_temperature(model_name):
            temp = temperature if temperature is not None else self.temperature
            request_kwargs["temperature"] = temp
        
        # Add tools if provided
        if tools:
            request_kwargs["tools"] = self._build_tools(tools)
            request_kwargs["tool_choice"] = "auto"
        
        # Add extra body params (e.g., reasoning effort)
        if extra_body:
            # For OpenRouter, pass extra params through
            for key, value in extra_body.items():
                request_kwargs[key] = value
        
        # Make the request
        self._log(f"Calling {model_name}...")
        start_time = time.time()
        
        try:
            response = completion(**request_kwargs)
            elapsed = time.time() - start_time
            self._log(f"Response received in {elapsed:.2f}s")
        except Exception as e:
            self._log(f"Error: {type(e).__name__}: {e}")
            raise LLMError(code="api_error", message=str(e)) from e
        
        # Parse response
        result = self._parse_response(response)
        
        # Update stats
        self._request_count += 1
        self._total_tokens += result.tokens.total
        self._total_cost += result.cost
        
        return result
    
    def ask(self, prompt: str, **kwargs) -> LLMResponse:
        """Simple ask method for single-turn queries.
        
        Args:
            prompt: The question/prompt
            **kwargs: Additional arguments for chat()
            
        Returns:
            LLMResponse
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        on_chunk: Optional[Any] = None,
        **kwargs,
    ) -> Generator[str, None, LLMResponse]:
        """Stream a chat response.
        
        Note: For simplicity, this currently uses non-streaming and yields full response.
        
        Args:
            messages: List of messages
            tools: Optional list of tools
            on_chunk: Optional callback for each chunk
            **kwargs: Additional arguments
            
        Yields:
            Text chunks
            
        Returns:
            Final LLMResponse
        """
        # For now, use non-streaming
        response = self.chat(messages, tools, **kwargs)
        
        if response.text:
            if on_chunk:
                on_chunk(response.text)
            yield response.text
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "request_count": self._request_count,
        }
    
    def close(self) -> None:
        """Close the client (no-op for litellm)."""
        pass
    
    def _log(self, msg: str) -> None:
        """Log to stderr."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [litellm] {msg}", file=sys.stderr, flush=True)


class LLMError(Exception):
    """LLM API error."""
    
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class CostLimitExceeded(LLMError):
    """Cost limit exceeded error."""
    
    def __init__(self, used: float, limit: float):
        self.used = used
        self.limit = limit
        super().__init__(
            code="cost_limit_exceeded",
            message=f"Cost limit exceeded: ${used:.4f} / ${limit:.4f}",
        )


# Alias for compatibility
LLM = LiteLLMClient
