import argparse
import time
import sys
import requests
import json
from typing import List, Dict

# --- CONFIGURATION ---
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"  # <--- REPLACE WITH YOUR BACKEND IP
POLL_INTERVAL = 1.0  # Seconds to wait when queue is empty

# build_prompt: helper to generate prompts given the message (system_instructions, user_message)
def build_prompt(tokenizer, messages: List[Dict[str, str]]):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Fallback for manual formatting
    parts = []
    for m in messages:
        parts.append(f"[{m.get('role', 'user')}] {m.get('content', '')}")
    parts.append("[assistant]")
    return "\n".join(parts)

def main():
    parser = argparse.ArgumentParser(description="GPU Worker for SubText")
    parser.add_argument("--model-path", default="/home/AI/qwen_7b_instruct")
    parser.add_argument("--backend", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="auto")
    # add --system arguments later in tasks 
    args = parser.parse_args()

    # --- LOAD MODEL ---
    print(f"⏳ Loading Model from {args.model_path}...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: Missing libraries.", file=sys.stderr)
        sys.exit(1)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"⚙️  Running on: {device.upper()}")
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model = model.to(device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    print(f"✅ Worker Online! Polling {args.backend}...")

    # --- WORKER LOOP --- 
    while True: 
        task_id = "UNKNOWN"
        try: 
            # Poll for work 
            try:
                response = requests.post(f"{args.backend}/queue/pop", timeout=5)
            except requests.exceptions.RequestException:
                print(f"❌ Cannot connect to Backend. Retrying in 5s...")
                time.sleep(5)
                continue
            
            if response.status_code == 404:
                time.sleep(POLL_INTERVAL)
                continue

            if response.status_code != 200:
                print(f"⚠️ Backend Error ({response.status_code}): {response.text}")
                time.sleep(POLL_INTERVAL)
                continue
            
            # Parse LLMTask Schema
            task = response.json()
            print(f"📦 RAW DATA FROM SERVER: {task}")
            task_id = task.get('id')

            if not task_id:
                print(f"❌ ERROR: Could not find 'id' in task keys: {task.keys()}")
                # Try 'task_id' just in case your schema is different
                task_id = task.get('task_id', "UNKNOWN")

            print(f"🚀 Processing Task {task_id}")

            system_instr = task.get('system_instruction', "You are a helpful assistant.")
            user_msg = task.get('user_message', "")

            print(f"🚀 Processing Task {task_id} ({task.get('domain', 'unknown')})")

            # Construct Messages Directly (system_instructions, user_message)
            # No reformatting done here, taken directly from backend logic
            messages = []
            if system_instr:
                messages.append({"role": "system", "content": system_instr})
            
            messages.append({"role": "user", "content": user_msg})

            prompt = build_prompt(tokenizer, messages)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Run 
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                )
            
            # Decode Output
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            result_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Submit Result
            requests.post(f"{args.backend}/queue/complete/{task_id}", json={
                "task_id": task_id,
                "result": result_text,
                "status": "completed"
            })
            print(f"✅ Finished {task_id}")

        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Backend. Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"❌ Worker Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()