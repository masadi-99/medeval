"""
Model providers for different LLM backends
"""

import asyncio
import time
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import openai

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class ModelResponse:
    """Standardized response format from any model provider"""
    content: str
    success: bool
    error: Optional[str] = None
    thinking_content: Optional[str] = None  # For models that support thinking mode
    metadata: Optional[Dict] = None


class ModelProvider(ABC):
    """Abstract base class for model providers"""
    
    @abstractmethod
    def query(self, prompt: str, **kwargs) -> ModelResponse:
        """Synchronous query to the model"""
        pass
    
    @abstractmethod
    async def query_async(self, prompt: str, request_id: str = None, **kwargs) -> ModelResponse:
        """Asynchronous query to the model"""
        pass
    
    @abstractmethod
    def supports_concurrent(self) -> bool:
        """Whether this provider supports concurrent requests"""
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", rate_limiter=None):
        self.client = openai.OpenAI(api_key=api_key)
        self.async_client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.rate_limiter = rate_limiter
    
    def query(self, prompt: str, **kwargs) -> ModelResponse:
        """Synchronous OpenAI query"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical expert providing diagnostic assessments."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', 0.1)
            )
            return ModelResponse(
                content=response.choices[0].message.content.strip(),
                success=True
            )
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def query_async(self, prompt: str, request_id: str = None, **kwargs) -> ModelResponse:
        """Asynchronous OpenAI query"""
        if self.rate_limiter:
            await self.rate_limiter.wait_if_needed()
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical expert providing diagnostic assessments."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', 0.1)
            )
            return ModelResponse(
                content=response.choices[0].message.content.strip(),
                success=True,
                metadata={'request_id': request_id}
            )
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error=str(e),
                metadata={'request_id': request_id}
            )
    
    def supports_concurrent(self) -> bool:
        return True


class HuggingFaceProvider(ModelProvider):
    """Hugging Face transformers provider for local inference"""
    
    def __init__(self, model_name: str, device: str = "auto", torch_dtype: str = "auto", 
                 thinking_mode: bool = True, max_length: int = 32768):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for HuggingFace models. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.thinking_mode = thinking_mode
        self.max_length = max_length
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        print(f"Model loaded successfully!")
        
        # Set generation parameters based on model type and mode
        self._set_generation_params()
    
    def _set_generation_params(self):
        """Set optimal generation parameters based on model and mode"""
        if "qwen3" in self.model_name.lower():
            if self.thinking_mode:
                # Qwen3 thinking mode parameters
                self.generation_params = {
                    'temperature': 0.6,
                    'top_p': 0.95,
                    'top_k': 20,
                    'do_sample': True,
                    'pad_token_id': self.tokenizer.eos_token_id
                }
            else:
                # Qwen3 non-thinking mode parameters
                self.generation_params = {
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 20,
                    'do_sample': True,
                    'pad_token_id': self.tokenizer.eos_token_id
                }
        else:
            # Default parameters for other models
            self.generation_params = {
                'temperature': 0.1,
                'top_p': 0.9,
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id
            }
    
    def _prepare_messages(self, prompt: str) -> List[Dict]:
        """Prepare messages in the format expected by the model"""
        return [
            {"role": "system", "content": "You are a medical expert providing diagnostic assessments."},
            {"role": "user", "content": prompt}
        ]
    
    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply the model's chat template"""
        if "qwen3" in self.model_name.lower():
            # Qwen3 specific chat template with thinking mode
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking_mode
            )
        else:
            # Standard chat template for other models
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    
    def _parse_response(self, generated_text: str, input_length: int) -> ModelResponse:
        """Parse the model response, handling thinking content if present"""
        # Extract only the generated part
        response_text = generated_text[input_length:]
        
        # For Qwen3, parse thinking content
        if "qwen3" in self.model_name.lower() and self.thinking_mode:
            thinking_content = ""
            actual_content = response_text
            
            # Look for thinking tags
            if "<think>" in response_text and "</think>" in response_text:
                import re
                think_pattern = r'<think>(.*?)</think>'
                think_matches = re.findall(think_pattern, response_text, re.DOTALL)
                if think_matches:
                    thinking_content = "\n".join(think_matches).strip()
                    # Remove thinking content from actual response
                    actual_content = re.sub(think_pattern, '', response_text, flags=re.DOTALL).strip()
            
            return ModelResponse(
                content=actual_content.strip(),
                success=True,
                thinking_content=thinking_content if thinking_content else None
            )
        else:
            return ModelResponse(
                content=response_text.strip(),
                success=True
            )
    
    def query(self, prompt: str, **kwargs) -> ModelResponse:
        """Synchronous Hugging Face query"""
        try:
            messages = self._prepare_messages(prompt)
            text = self._apply_chat_template(messages)
            
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_length = len(model_inputs.input_ids[0])
            
            # Generate
            generation_params = self.generation_params.copy()
            generation_params.update(kwargs)
            generation_params['max_new_tokens'] = min(
                kwargs.get('max_tokens', 512), 
                self.max_length - input_length
            )
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    **generation_params
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            return self._parse_response(generated_text, len(text))
            
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def query_async(self, prompt: str, request_id: str = None, **kwargs) -> ModelResponse:
        """Asynchronous query (runs sync query in thread pool)"""
        # For local models, we run the sync version in a thread pool
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, self.query, prompt, **kwargs)
            if response.metadata is None:
                response.metadata = {}
            response.metadata['request_id'] = request_id
            return response
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error=str(e),
                metadata={'request_id': request_id}
            )
    
    def supports_concurrent(self) -> bool:
        # Local models can handle concurrent requests through thread pool
        return True


class HuggingFaceAPIProvider(ModelProvider):
    """Hugging Face Inference API provider"""
    
    def __init__(self, model_name: str, api_token: str, thinking_mode: bool = True):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required for HuggingFace API. Install with: pip install requests")
        
        self.model_name = model_name
        self.api_token = api_token
        self.thinking_mode = thinking_mode
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
    
    def _prepare_payload(self, prompt: str, **kwargs) -> Dict:
        """Prepare API payload"""
        # Simulate chat format
        messages = [
            {"role": "system", "content": "You are a medical expert providing diagnostic assessments."},
            {"role": "user", "content": prompt}
        ]
        
        # Format as a conversation
        formatted_prompt = "System: You are a medical expert providing diagnostic assessments.\n"
        formatted_prompt += f"User: {prompt}\n"
        formatted_prompt += "Assistant:"
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": kwargs.get('max_tokens', 512),
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.8),
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        return payload
    
    def query(self, prompt: str, **kwargs) -> ModelResponse:
        """Synchronous HuggingFace API query"""
        try:
            import requests
            
            payload = self._prepare_payload(prompt, **kwargs)
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    return ModelResponse(
                        content=generated_text.strip(),
                        success=True
                    )
                else:
                    return ModelResponse(
                        content="",
                        success=False,
                        error="Unexpected API response format"
                    )
            else:
                return ModelResponse(
                    content="",
                    success=False,
                    error=f"API request failed: {response.status_code} - {response.text}"
                )
                
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def query_async(self, prompt: str, request_id: str = None, **kwargs) -> ModelResponse:
        """Asynchronous HuggingFace API query"""
        try:
            import aiohttp
            
            payload = self._prepare_payload(prompt, **kwargs)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get('generated_text', '')
                            return ModelResponse(
                                content=generated_text.strip(),
                                success=True,
                                metadata={'request_id': request_id}
                            )
                        else:
                            return ModelResponse(
                                content="",
                                success=False,
                                error="Unexpected API response format",
                                metadata={'request_id': request_id}
                            )
                    else:
                        error_text = await response.text()
                        return ModelResponse(
                            content="",
                            success=False,
                            error=f"API request failed: {response.status} - {error_text}",
                            metadata={'request_id': request_id}
                        )
                        
        except ImportError:
            # Fallback to sync version
            response = self.query(prompt, **kwargs)
            if response.metadata is None:
                response.metadata = {}
            response.metadata['request_id'] = request_id
            return response
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error=str(e),
                metadata={'request_id': request_id}
            )
    
    def supports_concurrent(self) -> bool:
        return True


def create_model_provider(provider_type: str, **kwargs) -> ModelProvider:
    """Factory function to create model providers"""
    
    if provider_type.lower() == "openai":
        return OpenAIProvider(
            api_key=kwargs.get('api_key'),
            model=kwargs.get('model', 'gpt-4o-mini'),
            rate_limiter=kwargs.get('rate_limiter')
        )
    
    elif provider_type.lower() == "huggingface":
        return HuggingFaceProvider(
            model_name=kwargs.get('model_name'),
            device=kwargs.get('device', 'auto'),
            torch_dtype=kwargs.get('torch_dtype', 'auto'),
            thinking_mode=kwargs.get('thinking_mode', True),
            max_length=kwargs.get('max_length', 32768)
        )
    
    elif provider_type.lower() == "huggingface_api":
        return HuggingFaceAPIProvider(
            model_name=kwargs.get('model_name'),
            api_token=kwargs.get('api_token'),
            thinking_mode=kwargs.get('thinking_mode', True)
        )
    
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


# Predefined model configurations
PREDEFINED_MODELS = {
    # OpenAI models
    'gpt-4o-mini': {'provider': 'openai', 'model': 'gpt-4o-mini'},
    'gpt-4o': {'provider': 'openai', 'model': 'gpt-4o'},
    'gpt-4-turbo': {'provider': 'openai', 'model': 'gpt-4-turbo'},
    'gpt-3.5-turbo': {'provider': 'openai', 'model': 'gpt-3.5-turbo'},
    
    # Hugging Face models (local)
    'qwen3-30b': {
        'provider': 'huggingface', 
        'model_name': 'Qwen/Qwen3-30B-A3B',
        'thinking_mode': True,
        'max_length': 32768
    },
    'qwen3-30b-no-thinking': {
        'provider': 'huggingface', 
        'model_name': 'Qwen/Qwen3-30B-A3B',
        'thinking_mode': False,
        'max_length': 32768
    },
    
    # Add more models as needed
    'llama3-8b': {
        'provider': 'huggingface',
        'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct'
    },
    'mistral-7b': {
        'provider': 'huggingface',
        'model_name': 'mistralai/Mistral-7B-Instruct-v0.3'
    }
} 