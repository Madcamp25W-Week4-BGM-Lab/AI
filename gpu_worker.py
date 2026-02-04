import argparse
import time
import sys
import requests
import json
from chat_qwen import load_model, generate_from_prompt, generate_from_task

# --- CONFIGURATION ---
DEFAULT_BACKEND_URL = "http://172.10.5.176"  # <--- REPLACE WITH YOUR BACKEND IP
POLL_INTERVAL = 1.0  # Seconds to wait when queue is empty

def main():
    parser = argparse.ArgumentParser(description="GPU Worker for SubText")
    parser.add_argument("--model-path", default="/home/qwen_7b_instruct")
    parser.add_argument("--backend", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="auto")
    # add --system arguments later in tasks 
    args = parser.parse_args()

    # --- LOAD MODEL ---
    print(f"⏳ Loading Model from {args.model_path}...")
    try:
        tokenizer, model, device = load_model(args.model_path, device=args.device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    print(f"⚙️  Running on: {device.upper()}")
    print(f"✅ Worker Online! Polling {args.backend}...")

    # --- WORKER LOOP --- 
    while True: 
        task_id = None
        stage = "poll"
        completed = False
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
            task_id = task.get("id")
            if not task_id:
                raise ValueError(f"Missing task.id in payload keys={list(task.keys())}")

            domain = task.get("domain", "unknown")
            print(f"🚀 Processing Task {task_id} (domain={domain})")

            system_instr = task.get('system_instruction', "You are a helpful assistant.")
            user_msg = task.get('user_message', "")

            # Use the raw prompt as-is (no formatting or prompt edits).
            stage = "generate"
            print(f"🧠 Generate start {task_id}")
            result_text = generate_from_prompt(
                tokenizer,
                model,
                user_msg,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
            )
            print(f"🧠 Generate done {task_id}")

            # Submit Result
            stage = "complete"
            requests.post(f"{args.backend}/queue/complete/{task_id}", json={"result": result_text})
            completed = True
            print(f"✅ Finished {task_id}")

        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Backend. Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            err_text = f"[ERROR][{stage}] {e}"
            print(f"❌ Worker Error: {err_text}")
            try:
                if task_id and not completed:
                    requests.post(f"{args.backend}/queue/complete/{task_id}", json={"result": err_text})
                    completed = True
            except Exception:
                pass
            time.sleep(1)

if __name__ == "__main__":
    main()
