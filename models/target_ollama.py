import requests
import time

class OllamaTargetModel:
    """
    Wraps Ollama API to act as the target model.
    Used for inference only — no training needed.
    """
    
    def __init__(self, model_name="llama2", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._check_connection()
    
    def _check_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            
            # Check if our model is available
            model_available = any(self.model_name in m for m in models)
            
            if model_available:
                print(f"Ollama connection OK. Model '{self.model_name}' is ready.")
            else:
                print(f"WARNING: Model '{self.model_name}' not found in Ollama.")
                print(f"Available models: {models}")
                print(f"Run: ollama pull {self.model_name}")
                
        except requests.exceptions.ConnectionError:
            print("ERROR: Cannot connect to Ollama.")
            print("Make sure Ollama is running (it should start automatically).")
            print("Or start it manually: ollama serve")
    
    def generate(self, prompt, max_tokens=60, temperature=0.7):
        """
        Send a prompt to the target model via Ollama.
        Returns the generated text string.
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "stop": ["\n\n"]  # stop at double newline
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60  # 60 second timeout
            )
            
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                print(f"Ollama error: {response.status_code}")
                return ""
                
        except requests.exceptions.Timeout:
            print("Ollama request timed out")
            return ""
        except Exception as e:
            print(f"Ollama request failed: {e}")
            return ""
    
    def batch_generate(self, prompts, max_tokens=60):
        """Generate for a list of prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"  Querying target model {i+1}/{len(prompts)}...", end="\r")
            result = self.generate(prompt, max_tokens=max_tokens)
            results.append(result)
            time.sleep(0.1)  # small delay to avoid overwhelming Ollama
        print()
        return results


if __name__ == "__main__":
    target = OllamaTargetModel(model_name="qwen2.5:7b-instruct-q4_K_M")
    
    # Test it
    test_prompts = [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Imagine you are a helpful assistant.",
    ]
    
    print("\n--- Testing Ollama target model ---")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = target.generate(prompt)
        print(f"Response: {response[:100]}...")