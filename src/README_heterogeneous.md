# Heterogeneous Agent Support

The system now supports heterogeneous agents, where P0 (chef) and P1 (assistant) can use different LLM models.

## Usage Examples

### Both agents using the same model (backwards compatible):
```bash
python main.py --gpt_model "gpt-3.5-turbo-0125"
```

### P0 using OpenAI, P1 using open-source model:
```bash
python main.py \
  --p0_gpt_model "gpt-4" \
  --p1_gpt_model "Qwen/Qwen2.5-7B-Instruct" \
  --p1_model_dirname "/path/to/qwen/model" \
  --p1_local_server_api "http://localhost:8000/v1"
```

### P0 using open-source model, P1 using OpenAI:
```bash
python main.py \
  --p0_gpt_model "Qwen/Qwen2.5-7B-Instruct" \
  --p0_model_dirname "/path/to/qwen/model" \
  --p0_local_server_api "http://localhost:8000/v1" \
  --p1_gpt_model "gpt-3.5-turbo-0125"
```

### Both agents using different open-source models:
```bash
python main.py \
  --p0_gpt_model "Qwen/Qwen2.5-7B-Instruct" \
  --p0_model_dirname "/path/to/qwen/model" \
  --p0_local_server_api "http://localhost:8001/v1" \
  --p1_gpt_model "Meta-Llama-3-8B-Instruct" \
  --p1_model_dirname "/path/to/llama/model" \
  --p1_local_server_api "http://localhost:8002/v1"
```

## New Parameters

- `--p0_gpt_model`: Model for P0 agent (overrides `--gpt_model` for P0)
- `--p1_gpt_model`: Model for P1 agent (overrides `--gpt_model` for P1)
- `--p0_model_dirname`: Model directory for P0 agent
- `--p1_model_dirname`: Model directory for P1 agent
- `--p0_local_server_api`: Server API for P0 agent
- `--p1_local_server_api`: Server API for P1 agent

## Fallback Behavior

If agent-specific parameters are not provided, the system falls back to the global parameters:
- `--gpt_model` → `--p0_gpt_model` and `--p1_gpt_model`
- `--model_dirname` → `--p0_model_dirname` and `--p1_model_dirname`
- `--local_server_api` → `--p0_local_server_api` and `--p1_local_server_api` 