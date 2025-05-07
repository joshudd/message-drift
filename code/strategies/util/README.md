# Experiment Runner for Message Propagation

This code simulates message propagation through a tree of agents, with each agent paraphrasing the message received from its parent. It then measures how much the message changes as it travels through the tree.

## Requirements

- Python 3.6+
- Required packages: torch, pandas, numpy, evaluate, graphviz, transformers, scipy, gensim, matplotlib, tqdm

## Usage

```bash
# Run with default gpt2 model (no Ollama required)
python exp.py

# Run with a specific model
python exp.py --model gpt2
python exp.py --model distilgpt2
```

## Ollama Integration

The code supports using Ollama for more powerful models. If you want to use Ollama:

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Start the Ollama service
3. Pull the model you want to use (e.g., `ollama pull mistral-small:24b-instruct-2501-q4_K_M`)
4. Run the experiment with the model name:

```bash
python exp.py --model mistral-small:24b-instruct-2501-q4_K_M
```

### Troubleshooting Ollama Errors

If you see the error `error calling ollama api: ollama api error: 404`, it usually means one of these issues:

1. **Ollama is not running**: Start Ollama before running the script
2. **Model is not available**: Pull the model first with `ollama pull <model-name>`
3. **Wrong model name**: Check the exact model name with `ollama list`

The script will automatically fall back to using gpt2 if Ollama is not available.

## Available Models

- `gpt2`: Small and fast but less accurate
- `distilgpt2`: Even smaller and faster
- `tiiuae/falcon-rw-1b`: Good balance of speed/quality
- `facebook/opt-125m`: Tiny but very fast
- `mistral-small:24b-instruct-2501-q4_K_M`: Best quality, 24b parameters, Apache license (requires Ollama)

## Output

The script creates a `results` directory with:
- SVG visualizations of the agent trees
- JSON files with all the data
- PNG graphs showing how metrics change with depth 