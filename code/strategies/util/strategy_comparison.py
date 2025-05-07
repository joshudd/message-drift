import numpy as np
import matplotlib.pyplot as plt
import os

def plot_strategy_comparison(results, filename="strategy_comparison.png"):
    """plot comparison of different strategies"""
    strategies = list(results.keys())
    
    # prepare data for plotting
    avg_rouge = [results[s]['avg_rouge'] for s in strategies]
    min_rouge = [results[s]['min_rouge'] for s in strategies]
    avg_cosine = [results[s]['avg_cosine'] for s in strategies]
    
    # create figure
    plt.figure(figsize=(12, 8))
    
    # bar plot of average rouge scores
    plt.subplot(2, 1, 1)
    x = np.arange(len(strategies))
    width = 0.35
    
    plt.bar(x - width/2, avg_rouge, width, label='avg rouge')
    plt.bar(x + width/2, min_rouge, width, label='min rouge')
    
    plt.xlabel('strategies')
    plt.ylabel('rouge score')
    plt.title('content preservation by strategy')
    plt.xticks(x, strategies)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # bar plot of semantic similarity
    plt.subplot(2, 1, 2)
    plt.bar(x, avg_cosine, width=0.6)
    
    plt.xlabel('strategies')
    plt.ylabel('cosine similarity')
    plt.title('semantic preservation by strategy')
    plt.xticks(x, strategies)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() 