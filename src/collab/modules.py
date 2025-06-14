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

from .utils import convert_messages_to_prompt, retry_with_exponential_backoff, is_openai_model
from .web_util import listen_to_server, output_to_port, username_record

cwd = os.getcwd()
# deepseek_key_file = os.path.join(cwd, "deepseek_key.txt")
openai_key_file = os.path.join(cwd, "openai_key.txt")

# Always import OpenAI for API compatibility (needed for both OpenAI and local servers)
import openai
from openai import OpenAI

with open(openai_key_file, "r") as f:
    context = f.read()
openai_key = context.split("\n")[0]

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
EMBEDDING_MODEL = "text-embedding-3-small"


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
    ):

        self.model = model
        self.model_dirname = model_dirname
        self.local_server_api = local_server_api
        self.retrival_method = retrival_method
        self.K = K

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

    def query_messages(self, rethink) -> list:
        sytem_message = [
            {
                "role": "system",
                "content": "You are an intelligent agent planner, you need to generate output and plan in the specified format according to the game rules and environmental status.",
            }
        ]
        query = sytem_message + [
            {
                "role": "user",
                "content": self.instruction_head_list[0]["content"]
                + "<input>\n"
                + self.current_user_message["content"],
            }
        ]
        return query

    @retry_with_exponential_backoff
    def query(
        self,
        key,
        proxy,
        stop=None,
        temperature=0.7,
        debug_mode="Y",
        trace=True,
        rethink=False,
        map="",
    ):
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
            with open(openai_key_file, "r") as f:
                context = f.read()
                openai_key = context.split("\n")[0]
            client = OpenAI(api_key=openai_key)
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
        if self.is_openai_model(self.model):
            print('🦋 model thats currently active:', self.model)
            with open(openai_key_file, "r") as f:
                context = f.read()
            key = context.split("\n")[0]
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


def if_two_sentence_similar_meaning(model, key, proxy, sentence1, sentence2):
    # If OpenAI is disabled, use a simple fallback
    key = ""
    if not is_openai_model(model):
        print('🦋 model thats currently active:', model)
        print("⚠️ Warning: OpenAI API is disabled. Using simple similarity check.")
        # Simple token-based similarity as fallback
        import re
        tokens1 = set(re.findall(r"\w+", sentence1.lower()))
        tokens2 = set(re.findall(r"\w+", sentence2.lower()))

        if not tokens1 or not tokens2:
            return False

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        jaccard_sim = intersection / union if union > 0 else 0
        return jaccard_sim > 0.8
    
    # If OpenAI is enabled, get the API key
    if is_openai_model(model):
        with open(openai_key_file, "r") as f:
            context = f.read()
        key = context.split("\n")[0]
        openai.api_key = key

    if sentence1 == "":
        sentence1 = " "
    if sentence2 == "":
        sentence2 = " "
    get_response = False
    while not get_response:
        try:
            client = OpenAI(api_key=key)
            response = client.embeddings.create(
                model=EMBEDDING_MODEL, input=[sentence1, sentence2]
            )
            get_response = True
        except Exception as e:
            rprint("[red][OPENAI ERROR][/red]:", e)
            time.sleep(1)
    embedding_1 = response.data[0].embedding
    embedding_2 = response.data[1].embedding
    score = 1 - spatial.distance.cosine(embedding_1, embedding_2)
    if score > 0.9:
        return True
    else:
        return False
