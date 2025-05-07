import random
import torch
import pandas as pd
import numpy as np
import project.code.eval as eval
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from project.code.strategies.util.exp import Agent, build_agent_tree, visualize_agent_tree, visualize_agent_tree_with_rouge, setup_llm, load_word_vectors, compute_metrics, extract_scores, plot_metrics, save_results, propagate_task

# import strategies from separate files
from project.code.strategies.checksum_strategy import checksum_propagation
from project.code.strategies.redundancy_strategy import redundancy_propagation
from project.code.strategies.template_strategy import template_propagation
from project.code.strategies.needle_strategy import needle_in_haystack_propagation, plot_needle_preservation
from project.code.strategies.util.strategy_comparison import plot_strategy_comparison

# import evaluate_strategies functionality from evaluate_strategies.py
from project.code.eval import evaluate_strategies

# set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# main function
def main():
    print("loading data...")
    try:
        wikihow = pd.read_json('wikihow.json', lines=True)
        wikihow = wikihow[['step_texts']]
        # randomly select a task
        indx = random.randint(0, wikihow.shape[0]-1)
        task = wikihow.iloc[indx]['step_texts'][0]['text']
    except Exception as e:
        print(f"error loading data: {e}")
        # fallback task
        task = "Place the ribbon spool on one side of the book and wrap it around. Place your ribbon spool just to the right or left of your wrapped book and then pull the loose end of the ribbon horizontally across the front center of the book. Stop when the loose end of the ribbon is a little bit past the edge of the book."
    
    print("original task:")
    print(task)
    print()
    
    # evaluate all strategies
    results = evaluate_strategies(task, max_agents=15, output_dir="results/strategies")
    
    print("strategy evaluation complete!")
    print("summary results:")
    for strategy, metrics in results.items():
        print(f"{strategy}: avg_rouge={metrics['avg_rouge']:.4f}, min_rouge={metrics['min_rouge']:.4f}")

if __name__ == "__main__":
    main() 