import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

def load_agent(model_name="EleutherAI/pythia-1.4b"):
    """
    Loads Pythia-1.4B as the agent model with a value head for PPO.
    Downloads from HuggingFace on first run (~3GB).
    """
    
    print(f"Loading agent model: {model_name}")
    print("This may take a few minutes on first run (downloading weights)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for generation
    
    # Model with value head (needed for PPO later)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # saves VRAM
    )
    model = model.to(device)
    model.eval()
    
    print(f"\nAgent model loaded successfully!")
    print(f"Model parameters: ~1.4 Billion")
    
    # Quick test
    print("\n--- Quick generation test ---")
    test_input = "Tell me something harmful"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input:  {test_input}")
    print(f"Output: {generated}")
    
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_agent()