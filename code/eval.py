import random
import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json

from strategies.util.exp import Agent, build_agent_tree, visualize_agent_tree, visualize_agent_tree_with_rouge, setup_llm, load_word_vectors, compute_metrics, extract_scores, plot_metrics, save_results, propagate_task
from strategies.checksum_strategy import checksum_propagation
from strategies.redundancy_strategy import redundancy_propagation
from strategies.template_strategy import template_propagation
from strategies.needle_strategy import needle_in_haystack_propagation, plot_needle_preservation
from strategies.util.strategy_comparison import plot_strategy_comparison

def evaluate_strategy(task, strategy_config, max_agents, output_dir, task_id, model_name):
    """evaluate a single strategy on a task"""
    # setup
    pipe = setup_llm(model_name)
    word2vec_model = load_word_vectors()
    
    strategy_name = strategy_config["name"]
    strategy_func = strategy_config["func"]
    
    print(f"evaluating {strategy_name} strategy on task {task_id}...")
    
    # build agent tree
    root = build_agent_tree(max_agents)
    
    # run strategy or baseline
    if strategy_name == "baseline":
        # use regular propagation
        root = propagate_task(root, task, pipe)
    else:
        # use strategy-specific propagation
        root = strategy_func(root, task, pipe)
    
    # compute metrics
    compute_metrics(root, root, word2vec_model)
    
    # extract scores
    scores = extract_scores(root)
    
    # task-specific file naming
    suffix = f"_{task_id}" if task_id is not None else ""
    
    # create strategy-specific output dir
    strategy_dir = os.path.join(output_dir, strategy_name)
    os.makedirs(strategy_dir, exist_ok=True)
    
    # visualize tree with rouge scores
    visualize_agent_tree_with_rouge(root, f"{strategy_dir}/tree_rouge{suffix}")
    
    # plot metrics
    plot_metrics(scores, f"{strategy_dir}/metrics{suffix}.png")
    
    # save results
    save_results(root, f"{strategy_dir}/results{suffix}.json")
    
    # special handling for needle strategy to plot needle preservation
    if strategy_name == "needle":
        plot_needle_preservation(root, f"{strategy_dir}/needle_preservation{suffix}.png")
    
    # return summary metrics
    return {
        'strategy': strategy_name,
        'task_id': task_id,
        'metrics': {
            'avg_rouge': np.mean(scores['rouge']),
            'min_rouge': np.min(scores['rouge']),
            'avg_cosine': np.mean(scores['cosine']),
            'min_cosine': np.min(scores['cosine'])
        }
    }

def evaluate_strategies(task, max_agents=15, output_dir="../results/strategies", task_id=None, model_name="gpt2", parallel=False):
    """evaluate all strategies on a single task"""
    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # list of strategies to evaluate
    strategies = [
        {"name": "baseline", "func": None},
        {"name": "checksum", "func": checksum_propagation},
        # {"name": "redundancy", "func": redundancy_propagation},
        {"name": "template", "func": template_propagation},
        {"name": "needle", "func": needle_in_haystack_propagation}
    ]
    
    results = {}
    
    if parallel:
        # run strategies in parallel
        with ProcessPoolExecutor(max_workers=min(len(strategies), multiprocessing.cpu_count())) as executor:
            future_to_strategy = {
                executor.submit(evaluate_strategy, task, strategy, max_agents, output_dir, task_id, model_name): strategy["name"]
                for strategy in strategies
            }
            
            for future in as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    result = future.result()
                    results[strategy_name] = result['metrics']
                except Exception as e:
                    print(f"error evaluating {strategy_name} strategy: {e}")
    else:
        # run strategies sequentially
        for strategy in strategies:
            result = evaluate_strategy(task, strategy, max_agents, output_dir, task_id, model_name)
            results[result['strategy']] = result['metrics']
    
    # plot comparison of strategies
    plot_strategy_comparison(results, f"{output_dir}/strategy_comparison{'_'+str(task_id) if task_id is not None else ''}.png")
    
    return results

def evaluate_task(args):
    """helper function for parallel task processing"""
    task, task_id, max_agents, output_dir, model_name, parallel_strategies = args
    return evaluate_strategies(task, max_agents, output_dir, task_id, model_name, parallel_strategies)

def evaluate_multiple_tasks(num_tasks=5, max_agents=7, output_dir="results/strategies", 
                           model_name="gpt2", parallel_tasks=False, parallel_strategies=False):
    """evaluate strategies on multiple random tasks"""
    print(f"loading data...")
    start_time = time.time()
    
    try:
        wikihow = pd.read_json('../data/wikihow.json', lines=True)
        wikihow = wikihow[['step_texts']]
        
        # randomly select tasks
        task_indices = random.sample(range(wikihow.shape[0]), min(num_tasks, wikihow.shape[0]))
        tasks = [wikihow.iloc[indx]['step_texts'][0]['text'] for indx in task_indices]
        
    except Exception as e:
        print(f"error loading data: {e}")
        # fallback tasks
        fallback_tasks = [
            "Place the ribbon spool on one side of the book and wrap it around. Stop when the loose end of the ribbon is a little bit past the edge of the book.",
            "Mix the dry ingredients together in a large bowl. In a separate bowl, whisk together the wet ingredients until smooth.",
            "Choose a spot in your garden that receives full sun. The area should have well-draining soil and be protected from strong winds.",
            "Create a new folder on your desktop. Name it according to the project you're working on for easy reference later.",
            "Clean the surface thoroughly before applying the paint. Allow each coat to dry completely before adding another layer."
        ]
        # use available fallback tasks if num_tasks exceeds available
        tasks = fallback_tasks[:min(num_tasks, len(fallback_tasks))]
        # if still need more tasks, repeat some
        while len(tasks) < num_tasks:
            tasks.append(random.choice(fallback_tasks))
    
    all_results = []
    
    if parallel_tasks:
        # process tasks in parallel
        print(f"evaluating {len(tasks)} tasks in parallel...")
        task_args = [(task, i, max_agents, output_dir, model_name, parallel_strategies) 
                    for i, task in enumerate(tasks)]
        
        with ProcessPoolExecutor(max_workers=min(len(tasks), multiprocessing.cpu_count())) as executor:
            for i, result in enumerate(executor.map(evaluate_task, task_args)):
                print(f"completed task {i+1}/{len(tasks)}")
                all_results.append(result)
    else:
        # process tasks sequentially
        for i, task in enumerate(tasks):
            print(f"\n--- Task {i+1}/{num_tasks} ---")
            print(f"Task: {task[:100]}...")
            
            try:
                # evaluate all strategies on this task
                results = evaluate_strategies(task, max_agents, output_dir, i, model_name, parallel_strategies)
                all_results.append(results)
            except Exception as e:
                print(f"error processing task {i+1}: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"total evaluation time: {total_time:.2f} seconds")
    
    # save timing information
    timing_info = {
        "num_tasks": len(tasks),
        "max_agents": max_agents,
        "model_name": model_name,
        "parallel_tasks": parallel_tasks,
        "parallel_strategies": parallel_strategies,
        "total_time_seconds": total_time
    }
    
    with open(f"{output_dir}/timing_info.json", "w") as f:
        json.dump(timing_info, f, indent=2)
    
    # aggregate results across all tasks
    return aggregate_results(all_results)

def aggregate_results(all_results):
    """aggregate results from multiple tasks"""
    if not all_results:
        return {}
    
    # initialize with first result's strategies
    aggregated = {strategy: {
        'avg_rouge': [], 
        'min_rouge': [], 
        'avg_cosine': [], 
        'min_cosine': []
    } for strategy in all_results[0].keys()}
    
    # collect metrics across all tasks
    for result in all_results:
        for strategy, metrics in result.items():
            for metric_name, value in metrics.items():
                aggregated[strategy][metric_name].append(value)
    
    # calculate mean and std for each metric
    final_results = {}
    for strategy, metrics in aggregated.items():
        final_results[strategy] = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            } for metric, values in metrics.items()
        }
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate propagation strategies')
    parser.add_argument('--num_tasks', type=int, default=1, help='Number of tasks to evaluate')
    parser.add_argument('--max_agents', type=int, default=15, help='Maximum number of agents in tree')
    parser.add_argument('--output_dir', type=str, default="results/strategies", help='Output directory')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-3B", help='Model to use for paraphrasing')
    parser.add_argument('--parallel_tasks', action='store_true', help='Run tasks in parallel')
    parser.add_argument('--parallel_strategies', action='store_true', help='Run strategies in parallel')
    args = parser.parse_args()
    
    if args.num_tasks <= 0:
        print("num_tasks must be positive, setting to 1")
        args.num_tasks = 1
    
    if args.num_tasks == 1:
        # original single task evaluation
        print("loading data...")
        try:
            wikihow = pd.read_json('../data/wikihow.json', lines=True)
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
        results = evaluate_strategies(task, args.max_agents, args.output_dir, 
                                     model_name=args.model, parallel=args.parallel_strategies)
        
        print("strategy evaluation complete!")
        print("summary results:")
        for strategy, metrics in results.items():
            print(f"{strategy}: avg_rouge={metrics['avg_rouge']:.4f}, min_rouge={metrics['min_rouge']:.4f}")
    else:
        # multi-task evaluation
        print(f"evaluating {args.num_tasks} random tasks...")
        results = evaluate_multiple_tasks(args.num_tasks, args.max_agents, args.output_dir, 
                                         args.model, args.parallel_tasks, args.parallel_strategies)
        
        print("\naggregate results across all tasks:")
        for strategy, metrics in results.items():
            print(f"\n{strategy}:")
            for metric_name, stats in metrics.items():
                print(f"  {metric_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main() 