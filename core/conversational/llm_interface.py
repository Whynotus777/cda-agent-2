"""
LLM Interface Module

Manages communication with local LLM (Ollama) for natural language understanding.
Handles model loading, querying, and response processing.
"""

import requests
import json
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Interface to Ollama LLM for natural language processing.

    Supports models like Llama 3 70B, potentially fine-tuned on EDA terminology,
    Verilog, SystemVerilog, and chip design documentation.
    """

    def __init__(
        self,
        model_name: str = "llama3:70b",
        ollama_host: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize LLM interface.

        Args:
            model_name: Name of Ollama model to use
            ollama_host: Ollama server endpoint
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation_history: List[Dict] = []
        self._cache: Dict[str, str] = {}
        self._cache_size = 64
        self._timeout_secs = 30
        self._retries = 3
        self._router_cfg: Dict = {}

        # Optional config overrides if available
        try:
            from utils import load_config
            cfg = load_config().get('llm', {})
            self._timeout_secs = int(cfg.get('timeout_secs', self._timeout_secs))
            self._retries = int(cfg.get('retries', self._retries))
            self._cache_size = int(cfg.get('cache_size', self._cache_size))
            self._router_cfg = cfg.get('router', {}) or {}
            # Backward-compat: also accept triage block as router settings
            if not self._router_cfg:
                self._router_cfg = cfg.get('triage', {}) or {}
        except Exception:
            pass

    def _verify_connection(self):
        """Verify that Ollama server is accessible"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama at {self.ollama_host}")

            # Check if desired model is available
            models = response.json().get('models', [])
            available_models = [m['name'] for m in models]

            if not any(self.model_name in m for m in available_models):
                logger.warning(
                    f"Model {self.model_name} not found in available models: {available_models}"
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_host}")

    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_history: bool = False,
        **kwargs
    ) -> str:
        """
        Send a query to the LLM and get response.

        Args:
            prompt: User prompt/question
            system_prompt: Optional system instruction
            use_history: Whether to include conversation history
            **kwargs: Additional parameters to override defaults

        Returns:
            LLM response as string
        """
        # Build messages
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if use_history:
            messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": prompt})

        # Select model via router if enabled
        # triage-first if enabled
        triage_cfg = self._router_cfg if self._router_cfg and 'models' in self._router_cfg else cfg.get('triage', {}) if 'cfg' in locals() else {}
        if triage_cfg and triage_cfg.get('enable', False) and kwargs.get('triage', True):
            triage_model = triage_cfg.get('models', {}).get('triage', 'llama3.2:3b')
            selected_model = triage_model
        else:
            selected_model = kwargs.get('model', self._route_model(prompt, context=kwargs.get('context_role')))

        # Prepare request payload
        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.temperature),
                "num_predict": kwargs.get('max_tokens', self.max_tokens),
            }
        }

        # Simple cache key
        cache_key = json.dumps({
            'model': self.model_name,
            'system': system_prompt or '',
            'prompt': prompt,
            'use_history': use_history,
        }, sort_keys=True)

        if cache_key in self._cache:
            return self._cache[cache_key]

        last_err = None
        for attempt in range(1, self._retries + 1):
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/chat",
                    json=payload,
                    timeout=self._timeout_secs
                )
                response.raise_for_status()
                result = response.json()
                assistant_message = result['message']['content']

                if use_history:
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": assistant_message})

                # Update cache (LRU-like by size limit)
                if len(self._cache) >= self._cache_size:
                    # Remove an arbitrary item (simple policy)
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = assistant_message

                # Fire-and-forget shadow call to orchestrator if enabled
                try:
                    self._shadow_orchestrator(messages)
                except Exception:
                    pass

                return assistant_message
            except requests.exceptions.RequestException as e:
                last_err = e
                if attempt < self._retries:
                    backoff = min(2 ** (attempt - 1), 8)
                    try:
                        import time
                        time.sleep(backoff)
                    except Exception:
                        pass
                else:
                    logger.error(f"LLM query failed after {self._retries} attempts: {e}")
                    raise RuntimeError(f"Failed to query LLM: {e}")

    def query_streaming(self, prompt: str, system_prompt: Optional[str] = None):
        """
        Stream LLM response token by token (for real-time UI updates).

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction

        Yields:
            Individual response tokens as they arrive
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
        }

        try:
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'message' in data:
                        content = data['message'].get('content', '')
                        if content:
                            yield content

        except requests.exceptions.RequestException as e:
            logger.error(f"Streaming query failed: {e}")
            raise RuntimeError(f"Failed to stream LLM response: {e}")

    def _approx_token_count(self, text: str) -> int:
        """Fast approximate token count (~4 chars/token heuristic)."""
        try:
            return max(1, int(len(text) / 4))
        except Exception:
            return len(text)

    def _route_model(self, prompt: str, context: Optional[str] = None) -> str:
        """Choose a model based on length thresholds and context role.

        Falls back to self.model_name if router is disabled/misconfigured.
        """
        try:
            if not self._router_cfg or not self._router_cfg.get('enable', False):
                return self.model_name

            # Prefer token thresholds
            tok_th = self._router_cfg.get('token_thresholds', {})
            triage_to_moderate = int(tok_th.get('triage_to_moderate', 200))
            moderate_to_complex = int(tok_th.get('moderate_to_complex', 600))

            # Fallback char thresholds
            char_th = self._router_cfg.get('char_thresholds', {})
            char_triage_to_moderate = int(char_th.get('triage_to_moderate', 100000))
            char_moderate_to_complex = int(char_th.get('moderate_to_complex', 200000))

            models = self._router_cfg.get('models', {})
            role = context or 'default'
            role_models = models.get(role, models.get('default', {}))

            tokens = self._approx_token_count(prompt)
            if tokens <= triage_to_moderate:
                return role_models.get('small', self.model_name)
            elif tokens <= moderate_to_complex:
                return role_models.get('medium', self.model_name)
            else:
                return role_models.get('large', self.model_name)
        except Exception:
            return self.model_name

    def _shadow_orchestrator(self, messages: List[Dict]):
        """Optionally send a background call to the 70B orchestrator for learning/orchestration.

        Non-blocking; errors are ignored.
        """
        cfg = self._router_cfg.get('shadow_orchestrator', {}) if self._router_cfg else {}
        if not cfg or not cfg.get('enable', False):
            return

        payload = {
            "model": cfg.get('model', 'llama3:70b'),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": int(cfg.get('max_tokens', 256)),
            }
        }

        try:
            import threading
            def _call():
                try:
                    requests.post(f"{self.ollama_host}/api/chat", json=payload, timeout=10)
                except Exception:
                    pass
            threading.Thread(target=_call, daemon=True).start()
        except Exception:
            pass

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def set_model(self, model_name: str):
        """
        Switch to a different Ollama model.

        Args:
            model_name: Name of the new model
        """
        self.model_name = model_name
        self.clear_history()
        logger.info(f"Switched to model: {model_name}")

    def get_available_models(self) -> List[str]:
        """
        Get list of available models in Ollama.

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get('models', [])
            return [m['name'] for m in models]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model list: {e}")
            return []

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for text (useful for semantic search).

        Args:
            text: Input text to embed

        Returns:
            Embedding vector
        """
        payload = {
            "model": self.model_name,
            "prompt": text
        }

        try:
            response = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get('embedding', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
