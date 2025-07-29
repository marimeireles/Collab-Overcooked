import os
import sys
import time
from typing import Union

import numpy as np
import pandas as pd
import tiktoken
from rich import print as rprint
from scipy import spatial
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from .utils import convert_messages_to_prompt, retry_with_exponential_backoff, is_openai_model
from .web_util import listen_to_server, output_to_port, username_record

cwd = os.getcwd()
# deepseek_key_file = os.path.join(cwd, "deepseek_key.txt")
openai_key_file = os.path.join(cwd, "openai_key.txt")

# Always import OpenAI for API compatibility (needed for both OpenAI and local servers)
import openai
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Initialize sentence transformer model globally
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Define context window limits for different models
CONTEXT_WINDOW_LIMITS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-o1mini": 128000,
    "text-davinci-003": 4096,
    "deepseek-reasoner": 32768,
    "deepseek-chat": 32768,
    "deepseek-ai/DeepSeek-R1": 32768,
    "deepseek-ai/DeepSeek-V3": 32768,
    "DeepSeek-R1": 32768,
    "claude3_sonnet": 200000,
    # HuggingFace model identifiers (as used in vLLM)
    "mistralai/Mistral-7B-Instruct-v0.1": 8192,
    "Qwen/Qwen2.5-7B-Instruct": 8192,
    "Qwen/Qwen2.5-14B-Instruct": 8192,
    "Qwen/Qwen2.5-32B-Instruct": 26417,
    "meta-llama/Llama-3-8B-Instruct": 8192,
    # Simplified model names
    "qwen2.5-7b-instruct": 8192,
    "qwen2.5-14b-instruct": 8192,
    "qwen2.5-32b-instruct": 26417,
    "llama3-8b-instruct": 8192,
    "mistral-7b-instruct-v0.1": 8192,
    # For local models, use a conservative default
    "default": 8192,
}

# Only load OpenAI key when needed, not at module import time
def load_openai_key():
    """Load OpenAI API key only when needed for OpenAI models."""
    try:
        with open(openai_key_file, "r") as f:
            context = f.read()
        return context.split("\n")[0]
    except FileNotFoundError:
        raise FileNotFoundError(f"OpenAI key file not found at {openai_key_file}. This is only needed for OpenAI models.")

# global statistics
statistics_dict = {
    "total_timestamp": [],
    "total_order_finished": [],
    "total_score": 0,
    "total_action_list": [[], []],
    "content": [],
}

# turn statistics
turn_statistics_dict = {
    "timestamp": 0,
    "order_list": [],
    "actions": [],
    "map": "",
    "statistical_data": {
        "score": 0,
        "communication": [
            {"call": 0, "turn": [], "token": []},
            {"call": 0, "turn": [], "token": []},
        ],
        "error": [
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
        ],
        "error_correction": [
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
        ],
    },
    "content": {
        "observation": [[], []],
        "reflection": [[], []],
        "content": [[], []],
        "action_list": [[], []],
        "original_log": "",
    },
}

# LLM models
tokenizer, model = None, None
# Refer to https://platform.openai.com/docs/models/overview
TOKEN_LIMIT_TABLE = {
    "text-davinci-003": 4080,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "llama3:70b-instruct-fp16": 4096,
}
sys.path.append(os.getcwd())


class Module(object):
    """
    This module is responsible for communicating with LLMs.
    """

    def __init__(
        self,
        role_messages,
        model="gpt-3.5-turbo-0301",
        model_dirname="~/",
        local_server_api="http://localhost:8000/v1",
        retrival_method="recent_k",
        K=3,
        temperature=0.7,
    ):

        self.model = model
        self.model_dirname = model_dirname
        self.local_server_api = local_server_api
        self.retrival_method = retrival_method
        self.K = K
        self.temperature = temperature

        self.chat_model = True if "gpt" in self.model else False
        self.instruction_head_list = role_messages
        # a dynamic changed dialog_history used for generating  different input for each failure
        self.dialog_history_list = []
        # save the dialog_history of meetting first failture
        self.dialog_history_list_storage = []
        self.current_user_message = None
        self.cache_list = None
        self.experience = []
        self.embedding = None
        self.current_timestep = None

    def load_embedding(self):
        df = pd.read_csv(os.getcwd() + "/data/embedding_" + self.name.lower() + ".csv")
        df["embedding"] = df.embedding.apply(eval).apply(np.array)
        self.embedding = df

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)

    def get_cache(self) -> list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K :]
            else:
                return []
        else:
            return None

    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for a given text.
        Uses a more accurate estimation based on model type and content analysis.
        """
        if not text:
            return 0
        
        # For very short texts, use a conservative estimate
        if len(text) < 20:
            return max(1, len(text.split()))
        
        # Try to use tiktoken for more accurate estimation when available
        try:
            if "gpt" in self.model.lower():
                import tiktoken
                if "gpt-4" in self.model:
                    encoding = tiktoken.encoding_for_model("gpt-4")
                elif "gpt-3.5" in self.model:
                    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                else:
                    encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
        except (ImportError, Exception):
            pass
        
        # For other models, use improved heuristics
        # Count words, punctuation, and special characters
        words = len(text.split())
        
        # Special tokens and formatting add overhead
        special_chars = text.count('\n') + text.count('\t') + text.count('  ')
        json_like_chars = text.count('{') + text.count('}') + text.count('[') + text.count(']')
        
        # More conservative estimation: 3.5 characters per token for structured content
        # 4.5 characters per token for natural language
        if json_like_chars > 0 or special_chars > len(text) * 0.1:
            # Structured content (JSON, code, etc.) - more tokens
            base_tokens = len(text) // 3
        else:
            # Natural language - fewer tokens
            base_tokens = len(text) // 4.5
        
        # Add overhead for special formatting
        overhead = special_chars + json_like_chars
        
        # Conservative estimate: add 20% safety margin
        estimated = int((base_tokens + overhead) * 1.2)
        
        # Ensure minimum based on word count
        word_based_estimate = int(words * 1.3)  # Conservative word-to-token ratio
        
        return max(estimated, word_based_estimate)

    def get_context_window_limit(self) -> int:
        """
        Get the context window limit for the current model.
        """
        # Normalize model name for comparison
        model_lower = self.model.lower()
        
        # Check if exact model name exists
        if self.model in CONTEXT_WINDOW_LIMITS:
            return CONTEXT_WINDOW_LIMITS[self.model]
        
        # Check for partial matches (e.g., for models with version numbers)
        for model_key in CONTEXT_WINDOW_LIMITS:
            if model_key in self.model:
                return CONTEXT_WINDOW_LIMITS[model_key]
        
        # Special detection for the user's specific models
        if "qwen" in model_lower:
            if any(size in model_lower for size in ["7b", "14b", "32b"]):
                return 32768  # All Qwen2.5 models have 32K context
        
        if "llama" in model_lower and "8b" in model_lower:
            return 8192  # Llama3 8B has 8K context
        
        if "mistral" in model_lower and "7b" in model_lower:
            return 8192  # Mistral 7B has 8K context
        
        # Special handling for local models (often have paths like "/path/to/model")
        if "/" in self.model:
            # Most local models have similar context windows to open source models
            if any(name in model_lower for name in ["qwen", "yi"]):
                return 32768  # Common context window for these models
            elif any(name in model_lower for name in ["llama", "mistral"]):
                return 8192  # Common context window for these models
            elif any(name in model_lower for name in ["gpt", "chat"]):
                return 8192  # Conservative estimate for GPT-like models
        
        # Default for unknown models
        print(f"⚠️  Unknown model '{self.model}', using default context window limit: {CONTEXT_WINDOW_LIMITS['default']}")
        return CONTEXT_WINDOW_LIMITS["default"]

    def truncate_conversation_content(self, content: str, max_tokens: int) -> str:
        """
        Truncate conversation content to fit within the context window.
        Implements a sliding window approach that keeps the most recent content.
        """
        estimated_tokens = self.estimate_token_count(content)
        
        if estimated_tokens <= max_tokens:
            return content
        
        print(f"⚠️  Content too long ({estimated_tokens} tokens), truncating to {max_tokens} tokens...")
        
        # Apply safety margin to account for estimation inaccuracies
        target_tokens = int(max_tokens * 0.9)  # 10% safety margin
        
        # Split content into lines to preserve structure
        lines = content.split('\n')
        
        # Try to preserve the most recent content (end of conversation)
        truncated_lines = []
        current_tokens = 0
        
        # Start from the end and work backwards
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            line_tokens = self.estimate_token_count(line + '\n')
            
            if current_tokens + line_tokens <= target_tokens:
                truncated_lines.insert(0, line)
                current_tokens += line_tokens
            else:
                break
        
        # If we got some lines, use them
        if truncated_lines:
            truncated_content = '\n'.join(truncated_lines)
        else:
            # Emergency fallback: take the last portion that fits
            char_limit = int(target_tokens * 3.5)  # Conservative char-to-token ratio
            truncated_content = content[-char_limit:]
        
        # Final validation and adjustment
        final_tokens = self.estimate_token_count(truncated_content)
        if final_tokens > target_tokens:
            # If still too long, do character-based truncation
            reduction_factor = target_tokens / final_tokens
            new_char_limit = int(len(truncated_content) * reduction_factor)
            truncated_content = truncated_content[-new_char_limit:]
            final_tokens = self.estimate_token_count(truncated_content)
        
        print(f"✅ Content truncated from {estimated_tokens} to {final_tokens} tokens (kept ~{len(truncated_lines)}/{len(lines)} lines)")
        return truncated_content

    def query_messages(self, rethink) -> list:
        sytem_message = [
            {
                "role": "system",
                "content": "You are an intelligent agent planner, you need to generate output and plan in the specified format according to the game rules and environmental status.",
            }
        ]
        
        # Get context window limit for this model
        context_limit = self.get_context_window_limit()
        
        # Reserve space for system message and response (more conservative)
        system_tokens = self.estimate_token_count(sytem_message[0]["content"])
        instruction_tokens = self.estimate_token_count(self.instruction_head_list[0]["content"])
        reserved_tokens = system_tokens + instruction_tokens + 1000  # 1000 tokens for response + overhead
        
        # Calculate available tokens for conversation content
        available_tokens = context_limit - reserved_tokens
        
        # Get original conversation content
        conversation_content = self.current_user_message["content"]
        original_tokens = self.estimate_token_count(conversation_content)
        
        # Debug output
        print(f"🔍 Context window info: Model={self.model}, Limit={context_limit}, Reserved={reserved_tokens}, Available={available_tokens}, Original={original_tokens}")
        
        # Truncate conversation content if necessary
        if available_tokens > 0:
            truncated_content = self.truncate_conversation_content(conversation_content, available_tokens)
        else:
            # If no space left, take minimal content
            print(f"⚠️  No space left for conversation content, taking minimal content")
            truncated_content = conversation_content[-500:]  # Take last 500 chars (very conservative)
        
        # Build the query
        query = sytem_message + [
            {
                "role": "user",
                "content": self.instruction_head_list[0]["content"]
                + "<input>\n"
                + truncated_content,
            }
        ]
        
        # Final validation: ensure the entire query is within context limits
        total_query_tokens = sum(self.estimate_token_count(msg["content"]) for msg in query)
        safety_margin = 200  # Extra safety margin
        
        if total_query_tokens > (context_limit - safety_margin):
            print(f"⚠️  Query too long ({total_query_tokens} tokens), performing emergency truncation...")
            
            # Emergency truncation: reduce conversation content more aggressively
            max_conversation_tokens = context_limit - reserved_tokens - safety_margin
            if max_conversation_tokens < 100:
                max_conversation_tokens = 100  # Minimum viable content
            
            truncated_content = self.truncate_conversation_content(conversation_content, max_conversation_tokens)
            
            # Rebuild query with emergency truncation
            query = sytem_message + [
                {
                    "role": "user",
                    "content": self.instruction_head_list[0]["content"]
                    + "<input>\n"
                    + truncated_content,
                }
            ]
            
            final_tokens = sum(self.estimate_token_count(msg["content"]) for msg in query)
            print(f"🔧 Emergency truncation: {total_query_tokens} → {final_tokens} tokens")
        
        return query

    @retry_with_exponential_backoff
    def query(
        self,
        key,
        proxy,
        stop=None,
        temperature=None,
        debug_mode="Y",
        trace=True,
        rethink=False,
        map="",
    ):
        # Use instance temperature if none provided
        if temperature is None:
            temperature = self.temperature
        
        # Check if human is playing
        if "human" in self.model:
            # Human model logic
            receiver = self.name
            if receiver == "Chef":
                receiver = "agent0"
            elif receiver == "Assistant":
                receiver = "agent1"
            else:
                raise ValueError("Invalid agent name!")

            human_message = self.current_user_message["content"]
            if "DO NOT COMMUNICATE WITH YOUR TEAMMATE" in human_message:
                human_message = human_message[
                    human_message.find("DO NOT COMMUNICATE WITH YOUR TEAMMATE :\n")
                    + len("DO NOT COMMUNICATE WITH YOUR TEAMMATE :\n") :
                ]
                human_message = human_message[
                    : human_message.find("Below are the failed and analysis history")
                ]
            response = output_to_port(
                receiver, human_message, map=map, recipe=recipe, error=error
            )
            encoder_name = "llama3"  # Use llama3 tokenizer for human models

        # If not human mode, then LLM mode
        elif "/" in self.model:  # This indicates a local model path
            # Prepare messages for the model
            messages = self.query_messages(rethink)

            # Initialize vLLM client (using OpenAI-compatible API format)
            client = OpenAI(
                api_key="not-needed",  # vLLM implements OpenAI API format
                base_url=self.local_server_api,
            )

            # Make the request to local vLLM server
            response = client.chat.completions.create(
                model=self.model,  # Use the model name directly
                messages=messages,
                temperature=temperature,
            )
            encoder_name = "llama3"  # Use llama3 tokenizer for local models

        # Finally check for OpenAI models
        # TODO: this needs to be expanded to contain more OpenAI models
        elif any(model in self.model for model in ["gpt-3.5", "gpt-4", "text-davinci"]):
            messages = self.query_messages(rethink)
            key = load_openai_key()
            openai.api_key = key
            client = OpenAI(api_key=key)
            if "gpt-3.5" in self.model or "gpt-4" in self.model:
                response = client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature
                )
                encoder_name = "gpt-3.5-turbo" if "gpt-3.5" in self.model else "gpt-4"
            else:  # text-davinci-003
                prompt = convert_messages_to_prompt(messages)
                response = client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    stop=stop,
                    temperature=temperature,
                    max_tokens=256,
                )
                encoder_name = "p50k_base"
            time.sleep(1)
        else:
            raise ValueError(f"Unsupported model type: {self.model}")

        rs = self.parse_response(response)

        # Count tokens based on model type
        if "gpt" in encoder_name:
            # Use tiktoken for GPT models
            encoding = tiktoken.encoding_for_model(encoder_name)
            tokens = encoding.encode(rs)
            token_count = len(tokens)
        else:
            # Use llama tokenizer for all other models (including local models via vLLM)
            tokenizer = AutoTokenizer.from_pretrained(
                "../lib/llama_tokenizer", local_files_only=True
            )
            tokens = tokenizer.encode(rs)
            token_count = len(tokens)

        return rs, token_count

    def parse_response(self, response):
        """
        Parse the response from different model types.
        Handles OpenAI models (GPT-3.5, GPT-4, text-davinci), local models via vLLM,
        and human models.
        """
        if self.model == "claude3_sonnet":
            return response["content"][0]["text"]
        elif self.model in ["text-davinci-003"]:
            return response["choices"][0]["text"]
        elif self.model in [
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo",
            "gpt-4o",
        ]:
            return response["choices"][0]["message"]["content"]
        elif self.model in [
            "gpt-4",
            "gpt-4-0314",
            "gpt-4o-2024-05-13",
            "gpt-4o",
            "gpt-o1mini",
        ]:
            return response["choices"][0]["content"]
        elif self.model in [
            "deepseek-reasoner",
            "deepseek-chat",
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-V3",
            "DeepSeek-R1",
        ]:
            return response.choices[0].message.content
        elif "human" in self.model:
            response_template = (
                "{role} analysis: [NOTHING]\n{role} plan: {plan}\n{role} say: {say}"
            )
            if response["agent"] == "agent1":
                role = "Assistant"
            elif response["agent"] == "agent0":
                role = "Chef"
            else:
                raise ValueError("Return invalide agent info!")
            response_template = response_template.replace("{role}", role)
            response_template = response_template.replace("{plan}", response["plan"])
            response_template = response_template.replace(
                "{say}", response["say"] if response["say"] != "" else "[NOTHING]"
            )
            return response_template
        elif "text-davinci" in self.model:
            return response.choices[0].text
        else:
            # For all other models (including local models via vLLM)
            return response.choices[0].message.content

    def restrict_dialogue(self):
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        limit = TOKEN_LIMIT_TABLE[self.model]
        print(f"Current token: {self.prompt_token_length}")
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f"Update token: {self.prompt_token_length}")

    def reset(self):
        self.dialog_history_list = []

    def get_top_k_similar_example(self, key, k=4):
        if k == 0:
            return ""

        prompt_begin_chef = "Here are few examples to teach you the usage of your skills, but these are just some examples, you need to flexibly apply your skills according to the specific environment.\
You should make plan for yourself in 'Chef plan', and make plan for assistant by saying to him.\n"
        prompt_begin_assistant = "Here are few examples to teach you the usage of your skills, but these are just some examples, you need to flexibly apply your skills according to the specific environment.\
If you do not know what to do, just ask chef to make a plan for you.\n"
        recipe = """<example_recipe>
Recipe: 
NAME:
onion_soup

INGREDIENTS:
chopped_onion (1)

COOKING STEPs:
1. Put 1 onion into chopping board directly to get the chopped_onion, you should wait for 3 STEPs.
2. Put 1 chopped_onion into pot directly, you should wait for 10 STEPs.
</example_recipe>

"""  # get embedding for current input
        # if user set up openai add a custom key, otherwise, use empty key
        key = ""
        if is_openai_model(self.model):
            print('🦋 model thats currently active:', self.model)
            key = load_openai_key()
            openai.api_key = key

        get_response = False
        input = self.current_user_message["content"]
        while not get_response:
            try:
                client = OpenAI(api_key=key)
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL, input=[input]
                )
                get_response = True
            except Exception as e:
                rprint("[red][OPENAI ERROR][/red]:", e)
                time.sleep(1)

        input_embedding = response.data[0].embedding
        if self.embedding is None:
            self.load_embedding()

        self.embedding["similarities"] = self.embedding.embedding.apply(
            lambda x: 1 - spatial.distance.cosine(x, input_embedding)
        )
        top_k_strings = self.embedding.sort_values(
            "similarities", ascending=False
        ).head(k)["text"]
        result = ""
        for t in top_k_strings:
            if t[0] == "\n":
                t = t[1:]
            result += f"<example>\n{t}\n</example>\n\n"
        if self.name == "Chef":
            result = prompt_begin_chef + result
        elif self.name == "Assistant":
            result = prompt_begin_assistant + result

        return result

# Load an open-source, pre-trained sentence-embedding model
# You can swap in any model from https://www.sbert.net/docs/pretrained_models.html
_MODEL_NAME = "all-mpnet-base-v2"
_model = SentenceTransformer(_MODEL_NAME)

def if_two_sentence_similar_meaning(
    sentence1: str,
    sentence2: str,
    threshold: float = 0.9
) -> bool:
    """
    Returns True if the semantic similarity between sentence1 and sentence2
    exceeds the given threshold, using an open-source SentenceTransformer model.
    """
    # Handle empty sentences, as in your original
    if not sentence1:
        sentence1 = " "
    if not sentence2:
        sentence2 = " "

    # Compute embeddings for both sentences
    embeddings = _model.encode([sentence1, sentence2], convert_to_tensor=False)

    # Cosine similarity: 1 - cosine distance
    score = 1.0 - cosine(embeddings[0], embeddings[1])

    # Ensure threshold is a float; fall back to default if conversion fails
    try:
        threshold_val = float(threshold)
    except (TypeError, ValueError):
        threshold_val = 0.9  # default fallback

    return score > threshold_val
