# Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents

We propose a new LLM-powered Multi-Agent System (LLM-MAS) benchmark, Collab-Overcooked, built on the popular Overcooked-AI game with more applicable and challenging tasks in interactive environments. Collab-Overcooked extends existing benchmarks from two novel perspectives. First, it provides a multi-agent framework supporting diverse tasks and objectives and encourages collaboration through natural language communication. Second, it introduces a spectrum of process-oriented evaluation metrics to assess the fine-grained collaboration capabilities of different LLM agents, a dimension often overlooked in prior work.

## Getting Started

### Install
We recommend using the anaconda management environment. Python 3.8 is recommended for this project.  
- Install requirements
    - Directly do:
        ```
        conda create -n collab-overcooked python=3.8
        conda activate collab-overcooked

        pip install -r requirements.txt
        conda install mpi4py==3.1.4  # pip install often fails
        ```

- Install the game environment `overcooked_ai` locally.
    ```
    cd ./lib/overcooked_ai
    pip install -e .
    ```
    Notes : Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game Overcooked. Please refer to https://github.com/HumanCompatibleAI/overcooked_ai

### Quick Test
The easiest way to test whether the environment is installed correctly is to use gpt-3.5-turbo to test after filling in the openai api secret key. 
- Fill in the OpenAI API key at "Collab-Overcooked/src/openai_key.txt"
- Run the following commands
  ```
  cd Collab-Overcooked/src
  python main.py --horizon 3 --order boiled_egg
  ```
If you can output the environmental visualization map normally, the agents' normal output content, and run through 3 time steps without any errors, then your environment and agent configuration are successful. This will take you about 1.5 minutes, depending on the speed of your connection to OpenAI.

### Configure the Open-source LLMs
TODO

### Evaluation
TODO

## Modify the Environment
TODO

## Q&A
TODO

