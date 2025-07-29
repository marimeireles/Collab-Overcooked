#!/usr/bin/env python3

# Simple test script for context window management

import sys
import os
sys.path.append('src')

from collab.modules import Module, CONTEXT_WINDOW_LIMITS

def test_context_window_management():
    print("Testing context window management...")
    
    # Test the models based on the user's actual setup
    # From the shell script, models are stored in HuggingFace cache and referenced by HF names
    test_models = [
        # HuggingFace model identifiers (as used in vLLM)
        # "mistralai/Mistral-7B-Instruct-v0.1",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "Qwen/Qwen2.5-14B-Instruct", 
        # "Qwen/Qwen2.5-32B-Instruct",
        # "meta-llama/Llama-3-8B-Instruct",
        # Also test simplified versions that might be used
        # "mistral-7b-instruct-v0.1",
        "qwen2.5-7b-instruct",
        # "qwen2.5-14b-instruct",
        # "qwen2.5-32b-instruct",
        # "llama3-8b-instruct",
    ]
    
    print(f"Available context window limits: {list(CONTEXT_WINDOW_LIMITS.keys())}")
    print(f"Based on your shell script, models are stored in: /nas/ucb/marimeireles/cache/hub")
    
    for model in test_models:
        print(f"\n--- Testing model: {model} ---")
        
        # Create a module instance
        module = Module(
            role_messages=[{"role": "system", "content": "Test system message"}],
            model=model
        )
        
        # Test context window limit detection
        limit = module.get_context_window_limit()
        print(f"Context window limit: {limit}")
        
        # Test token estimation
        test_text = "This is a test message with some content. " * 100
        estimated_tokens = module.estimate_token_count(test_text)
        print(f"Estimated tokens for test text: {estimated_tokens}")
        
        # Test truncation with a realistic scenario
        max_tokens = 1000
        truncated = module.truncate_conversation_content(test_text, max_tokens)
        truncated_tokens = module.estimate_token_count(truncated)
        print(f"Truncated tokens: {truncated_tokens} (target: {max_tokens})")
        
        # Test with very large content (simulate the actual overflow issue)
        # This simulates the 32828 token issue from the error log
        large_content = "Long conversation history with game state and actions. " * 600  # Should exceed 32K context
        large_tokens = module.estimate_token_count(large_content)
        print(f"Large content tokens: {large_tokens}")
        
        # Simulate the full query_messages flow
        system_instruction = "You are an intelligent agent planner for the Overcooked game. " * 20
        module.instruction_head_list = [{"role": "system", "content": system_instruction}]
        module.current_user_message = {"role": "user", "content": large_content}
        
        try:
            query = module.query_messages(rethink=False)
            final_content = query[1]["content"]
            final_tokens = module.estimate_token_count(final_content)
            print(f"Final query tokens: {final_tokens}")
            
            if final_tokens > limit:
                print(f"❌ ERROR: Final tokens ({final_tokens}) exceed limit ({limit})")
            else:
                print(f"✅ SUCCESS: Final tokens ({final_tokens}) within limit ({limit})")
                
        except Exception as e:
            print(f"❌ ERROR in query_messages: {e}")

if __name__ == "__main__":
    test_context_window_management() 