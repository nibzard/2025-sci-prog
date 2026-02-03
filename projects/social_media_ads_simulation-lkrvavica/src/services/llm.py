
import os
import requests
import json
from typing import List, Optional, Dict, Any, Union

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from src.core.config import config

class LLMService:
    def __init__(self):
        # We can initialize clients here if needed, but for now they are stateless/request-based
        pass

    @staticmethod
    def clean_json_response(response_str: str) -> str:
        """Clean up markdown blocks and whitespace from LLM response."""
        if not response_str:
            return "{}"
        
        cleaned = response_str.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline:].strip()
            
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
                
        if not cleaned or not (cleaned.startswith("{") and cleaned.endswith("}")):
            return "{}"
            
        return cleaned

    def query(self, prompt: str, backend: str = "openai", model_name: Optional[str] = None, json_mode: bool = True) -> str:
        """
        Unified interface to query different LLM backends.
        """
        # Resolve defaults
        if not model_name:
            if backend == "openai":
                model_name = config.openai_model
            elif backend == "google":
                model_name = config.google_model
            elif backend == "local":
                model_name = config.local_model
            else:
                model_name = "gpt-4o"

        if backend == "openai":
            return self._query_openai(prompt, model_name, json_mode)
        elif backend == "google":
            return self._query_google(prompt, model_name, json_mode)
        elif backend == "local":
            return self._query_local(prompt, model_name, json_mode)
        else:
            return "{}" if json_mode else ""

    def _query_openai(self, prompt: str, model_name: str, json_mode: bool) -> str:
        key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key: 
            print("ERROR: Missing OpenAI API Key")
            return "{}" if json_mode else ""
            
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return self.clean_json_response(content) if json_mode else content.strip()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"DEBUG: OpenAI Query Failed: {e}")
            return "{}" if json_mode else ""

    def _query_google(self, prompt: str, model_name: str, json_mode: bool) -> str:
        key = config.google_api_key or os.getenv("GOOGLE_API_KEY")
        if not key: 
            print("ERROR: Missing Google API Key")
            return "{}" if json_mode else ""
            
        genai.configure(api_key=key)
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            # Google sometimes blocks content, check parts
            return self.clean_json_response(response.text) if json_mode else response.text.strip()
        except Exception as e:
            print(f"DEBUG: Google Query Failed: {e}")
            return "{}" if json_mode else ""

    def _query_local(self, prompt: str, model_name: str, json_mode: bool) -> str:
        url = f"{config.local_llm_url}/api/generate"
        payload = {"model": model_name, "prompt": prompt, "stream": False}
        if json_mode:
            payload["format"] = "json"
            
        try:
            r = requests.post(url, json=payload)
            r.raise_for_status()
            content = r.json().get("response", "{}")
            return self.clean_json_response(content) if json_mode else content.strip()
        except Exception as e:
            print(f"DEBUG: Local Query Failed: {e}")
            return "{}" if json_mode else ""

    def fetch_available_models(self, backend: str) -> List[str]:
        models = []
        if backend == "google":
            key = config.google_api_key or os.getenv("GOOGLE_API_KEY")
            if key:
                try:
                    genai.configure(api_key=key)
                    all_models = genai.list_models()
                    models = [m.name.replace("models/", "") for m in all_models if 'generateContent' in m.supported_generation_methods]
                except Exception: models = ["gemini-1.5-flash", "gemini-pro"]
        
        elif backend == "openai":
            key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if key:
                try:
                    headers = {"Authorization": f"Bearer {key}"}
                    r = requests.get("https://api.openai.com/v1/models", headers=headers)
                    if r.status_code == 200:
                        models = [m["id"] for m in r.json().get("data", []) if "gpt" in m["id"]]
                        models.sort()
                except Exception: models = ["gpt-4o", "gpt-4-turbo"]
        
        elif backend == "local":
            url = f"{config.local_llm_url}/api/tags"
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
            except Exception: models = ["llama3", "mistral"]
            
        return models if models else ["default"]

# Singleton instance
llm_client = LLMService()
