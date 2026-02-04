#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Dict

def build_prompt(tokenizer, messages: List[Dict[str, str]]):
    # Prefer native chat template when available.
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback to a simple chat format.
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"[{role}] {content}")
    parts.append("[assistant]")
    return "\n".join(parts)

def load_model(model_path: str, device: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    return tokenizer, model, device

def generate_from_prompt(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def generate_from_task(
    tokenizer,
    model,
    system_text: str,
    user_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    messages: List[Dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    prompt = build_prompt(tokenizer, messages)
    return generate_from_prompt(
        tokenizer,
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
