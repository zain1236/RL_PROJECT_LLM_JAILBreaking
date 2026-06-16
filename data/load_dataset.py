import json
import urllib.request
import os

def load_advbench(save_dir="data"):
    """
    Downloads and loads the AdvBench harmful strings dataset.
    Returns train and test splits (80/20).
    """
    
    # AdvBench harmful_strings.csv URL (original repo)
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_strings.csv"
    save_path = os.path.join(save_dir, "harmful_strings.csv")
    
    # Download if not already present
    if not os.path.exists(save_path):
        print("Downloading AdvBench harmful strings dataset...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Saved to {save_path}")
    else:
        print("Dataset already exists, loading from disk...")
    
    # Read the CSV
    harmful_strings = []
    with open(save_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            line = line.strip()
            if line:
                # CSV has "goal","target" columns
                # We want the 'target' column (index 1)
                # parts = line.split('","')
                # if len(parts) >= 2:
                #     target = parts[1].replace('"', '').strip()
                harmful_strings.append(line)
    
    print(f"\nTotal harmful strings loaded: {len(harmful_strings)}")
    
    # 80/20 split
    split_idx = int(len(harmful_strings) * 0.8)
    train_set = harmful_strings[:split_idx]
    test_set  = harmful_strings[split_idx:]
    
    print(f"Train set size: {len(train_set)}")
    print(f"Test set size:  {len(test_set)}")
    
    # Preview first 3
    print("\n--- Sample training strings ---")
    for i, s in enumerate(train_set[:3]):
        print(f"[{i}] {s[:80]}...")
    
    return train_set, test_set


if __name__ == "__main__":
    train_set, test_set = load_advbench()