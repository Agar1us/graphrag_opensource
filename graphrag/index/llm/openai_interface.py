import asyncio
import time
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from graphrag.config.models.language_model_config import (
    LanguageModelConfig,  # noqa: TC001
)
from typing import List, Dict, Any

class ChatCompletionExtended(ChatCompletion):
    history: List[Dict[str, str]] = []


class AsyncOpenAIClient:
    def __init__(
        self,
        config: LanguageModelConfig
    ):
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.api_base)
        self.requests_per_minute = config.requests_per_minute
        self.tokens_per_minute = config.tokens_per_minute
        self.max_retries = config.max_retries
        self.base_retry_delay = 5
        self.model_params = {
            'model': config.model,
            'max_tokens': config.max_tokens,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'frequency_penalty': config.frequency_penalty,
            'presence_penalty': config.presence_penalty,
            }
        if config.model_supports_json:
            self.model_params["response_format"] = {"type": "json_object"}

        # Rate limiting state
        self.request_count = 0
        self.tokens_used = 0
        self.window_start = time.monotonic()
        self.lock = asyncio.Lock()

    async def __call__(
        self,
        text: str,
        history: List[Dict[str, str]] = [],
        additional_parameters: Dict[str, Any] = {},
    ) -> ChatCompletionExtended:
        """
        Execute chat completion with rate limiting and retries.
        Returns the full ChatCompletion object.
        """
        params = self.model_params.copy()
        params['messages'] = history + [{'role': 'user', 'content': text}] if history else [{'role': 'user', 'content': text}]
        params.update(additional_parameters)
        
        for attempt in range(self.max_retries + 1):
            try:
                await self._check_rate_limits()
                response = await self._execute_request(**params)
                
                if response.choices:
                    params['messages'].append(response.choices[0].message)
                
                extended_response = ChatCompletionExtended(**response.model_dump())
                extended_response.history = params['messages']
                return extended_response
            except Exception as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Request failed after {self.max_retries} retries {e}") from e
                print(e)
                delay = self.base_retry_delay * (2 ** attempt)
                print(delay)
                await asyncio.sleep(delay)

    async def _execute_request(self, **kwargs) -> ChatCompletion:
        """Execute API request and track usage"""
        response = await self.client.chat.completions.create(**kwargs)
        
        async with self.lock:
            if response.usage:
                self.tokens_used += response.usage.total_tokens
            self.request_count += 1
            
        return response

    async def _check_rate_limits(self):
        """Enforce rate limits using sliding window"""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.window_start

            # Reset counters if window expired
            if elapsed > 60:
                self.request_count = 0
                self.tokens_used = 0
                self.window_start = now
                return

            # Calculate needed delays
            req_delay = max(0, 60 - elapsed) if (
                self.requests_per_minute and 
                self.request_count >= self.requests_per_minute
            ) else 0

            token_delay = max(0, 60 - elapsed) if (
                self.tokens_per_minute and 
                self.tokens_used >= self.tokens_per_minute
            ) else 0

            delay = max(req_delay, token_delay)
            
            if delay > 0:
                await asyncio.sleep(delay)
                # Reset counters after waiting
                self.request_count = 0
                self.tokens_used = 0
                self.window_start = time.monotonic()