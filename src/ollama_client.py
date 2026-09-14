import requests
import json
from typing import List, Dict, Any, Optional

OLLAMA_BASE_URL = "http://localhost:11434"

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def get_models(self) -> List[Dict[str, Any]]:
        """Fetch list of available models from local Ollama instance (excluding embedding-only models)."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                chat_models = []
                for m in models:
                    name = m.get("name", "")
                    caps = m.get("capabilities", [])
                    # Exclude embedding-only models
                    if "embed" in name.lower() or "bert" in m.get("details", {}).get("family", "").lower():
                        continue
                    if caps and "completion" not in caps and "generate" not in caps:
                        continue
                    chat_models.append(m)
                return chat_models
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
        return []

    def get_embedding(self, text: str, model: str = "nomic-embed-text") -> List[float]:
        """Generate embedding vector for a given text using Ollama."""
        try:
            # Truncate text to avoid token limits on embedding models (nomic-embed-text max context ~2048 tokens)
            safe_text = text[:3000].strip()
            if not safe_text:
                return [0.0] * 768

            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": safe_text},
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json().get("embedding", [])
            else:
                print(f"Embedding API error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"Error generating embedding: {e}")
        return []

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = "llama3.1:8b",
        temperature: float = 0.2,
        stream: bool = False
    ) -> str:
        """Generate LLM response using local model."""
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "options": {
                "temperature": temperature,
                "num_predict": 1024,
            },
            "stream": False
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
            else:
                return f"Error from Ollama ({resp.status_code}): {resp.text}"
        except Exception as e:
            return f"Failed to connect to local Ollama model: {str(e)}"
