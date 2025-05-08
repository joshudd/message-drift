import random
import torch
import pandas as pd
import numpy as np
import evaluate
from graphviz import Digraph
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from scipy.spatial.distance import cosine
import gensim.downloader as api
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# agent class for the tree structure
class Agent:
    def __init__(self, id, parent=None):
        self.id = id
        self.parent = parent
        self.children = []
        self.message = ""
        self.length_ratio = None
        self.rouge_score = None
        self.cosine_similarity = None

# build the agent tree with random number of children
def build_agent_tree(max_agents):
    if max_agents <= 0:
        return None

    agent_count = 1
    root = Agent(agent_count)
    agents = [root]
    index = 0

    while agent_count < max_agents and index < len(agents):
        parent_agent = agents[index]
        num_children = min(random.randint(1, 3), max_agents - agent_count)

        for _ in range(num_children):
            agent_count += 1
            new_agent = Agent(agent_count, parent=parent_agent)
            parent_agent.children.append(new_agent)
            agents.append(new_agent)
            if agent_count >= max_agents:
                break

        index += 1

    return root

# visualize the agent tree
def visualize_agent_tree(root, filename="agent_tree"):
    dot = Digraph()

    def add_nodes_edges(agent):
        dot.node(str(agent.id))
        for child in agent.children:
            dot.node(str(child.id))
            dot.edge(str(agent.id), str(child.id))
            add_nodes_edges(child)

    add_nodes_edges(root)
    dot.render(filename, format="svg", cleanup=True)
    return dot

# visualize agent tree with rouge scores
def visualize_agent_tree_with_rouge(root, filename="agent_tree_rouge"):
    dot = Digraph()

    def add_nodes_edges(agent):
        if agent.rouge_score is not None:
            rouge_val = agent.rouge_score
            color = f"#{int(255 - (rouge_val * 255)):02x}{int(rouge_val * 255):02x}00"
            dot.node(str(agent.id), style="filled", fillcolor=color)
        else:
            dot.node(str(agent.id))

        for child in agent.children:
            if child.rouge_score is not None:
                rouge_val = child.rouge_score
                color = f"#{int(255 - (rouge_val * 255)):02x}{int(rouge_val * 255):02x}00"
                dot.node(str(child.id), style="filled", fillcolor=color)
            else:
                dot.node(str(child.id))
            dot.edge(str(agent.id), str(child.id))
            add_nodes_edges(child)

    add_nodes_edges(root)
    dot.render(filename, format="svg", cleanup=True)
    return dot

# load mistral model for paraphrasing (or use a smaller model if memory constrained)
def setup_llm(model_name):
    """setup language model for paraphrasing tasks"""
    try:
        print(f"loading {model_name} model for paraphrasing...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            return_full_text=False
        )
        return pipe
    except Exception as e:
        print(f"error loading {model_name} model: {e}")
        
        # fallback to an even simpler function that just adds some variation
        print("falling back to simple string modification function...")
        
        # create a simple function that mimics the pipeline interface but just modifies the text slightly
        def simple_paraphrase(prompt):
            # extract input text from prompt
            if "[INST]" in prompt and "[/INST]" in prompt:
                input_text = prompt.split("[INST]")[1].split("[/INST]")[0].strip()
                # remove the instruction part
                if "Please paraphrase" in input_text:
                    input_text = input_text.split("\n\n")[1].strip()
            else:
                input_text = prompt
                
            # apply simple modifications (swap some words, change word order slightly)
            import random
            words = input_text.split()
            
            # randomly swap word positions
            if len(words) > 5:
                for _ in range(min(3, len(words) // 5)):
                    i, j = random.sample(range(len(words)), 2)
                    words[i], words[j] = words[j], words[i]
            
            # randomly insert some filler words
            fillers = ["basically", "essentially", "generally", "in other words", "simply put"]
            if len(words) > 7 and random.random() > 0.5:
                insert_pos = random.randint(0, len(words) - 1)
                words.insert(insert_pos, random.choice(fillers))
            
            # randomly replace some connecting words
            replacements = {
                "and": ["as well as", "plus", "along with"],
                "but": ["however", "yet", "nevertheless"],
                "if": ["when", "provided that", "assuming"],
                "then": ["next", "after that", "subsequently"],
                "also": ["furthermore", "moreover", "additionally"]
            }
            
            for i, word in enumerate(words):
                if word.lower() in replacements and random.random() > 0.7:
                    words[i] = random.choice(replacements[word.lower()])
            
            result = " ".join(words)
            
            # return in the same format as the pipeline would
            return [{"generated_text": f"Answer: {result}"}]
        
        return simple_paraphrase

# propagate message through the agent tree
def propagate_task(root, task, pipe):
    root.message = task
    
    def propagate(agent):
        print(f"Agent {agent.id} got the message.")
        if not agent.children:
            return
            
        for child in agent.children:
            if agent.message:
                # format prompt for the LLM
                prompt = f"""<s>[INST] Please paraphrase the following instructional message using your own words while preserving all important details:

{agent.message}

[/INST]"""
                try:
                    # check if pipe is a callable function (our fallback) or an actual pipeline
                    if callable(pipe) and not hasattr(pipe, 'model'):
                        response = pipe(prompt)
                    else:
                        response = pipe(prompt)
                        
                    # extract answer from the response
                    answer = response[0]["generated_text"]
                    
                    # sometimes the model adds "Answer:" prefix, remove if present
                    if answer.startswith("Answer:"):
                        answer = answer[7:].strip()
                        
                    child.message = answer
                except Exception as e:
                    print(f"Error generating response: {e}")
                    # fallback to simple copying with minor changes
                    import random
                    words = agent.message.split()
                    if len(words) > 10:
                        # remove a random word
                        del words[random.randint(0, len(words)-1)]
                    child.message = " ".join(words)
                
            propagate(child)
            
    propagate(root)
    return root

# load word vectors for semantic similarity
def load_word_vectors():
    try:
        word2vec_model = api.load("word2vec-google-news-300")
        return word2vec_model
    except Exception as e:
        print(f"Error loading word vectors: {e}")
        return None

# convert sentence to vector
def sentence_to_vector(sentence, model):
    """Convert a sentence into a vector by averaging its word embeddings."""
    if model is None:
        return np.zeros(300)  # fallback if model not loaded
        
    words = sentence.lower().split()
    word_vectors = [model[word] for word in words if word in model]

    if not word_vectors:
        return np.zeros(model.vector_size)

    return np.mean(word_vectors, axis=0)  # average word vectors

# compute cosine similarity
def cosine_similarity_score(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    try:
        # handle cases where vectors are all zeros
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        return 1 - cosine(vec1, vec2)  # cosine similarity
    except Exception as e:
        # handle any numerical errors
        print(f"Error computing cosine similarity: {e}")
        return 0.0

# compute metrics for all agents
def compute_metrics(agent, root, word2vec_model):
    # length ratio
    agent.length_ratio = len(agent.message) / len(root.message) if root.message else 0
    
    # rouge score
    rouge = evaluate.load("rouge")
    if agent.message and root.message:
        agent.rouge_score = rouge.compute(
            predictions=[agent.message], 
            references=[root.message]
        )['rouge1']
    else:
        agent.rouge_score = 0
    
    # word vector similarity
    if word2vec_model is not None and agent.message and root.message:
        vector1 = sentence_to_vector(agent.message, word2vec_model)
        vector2 = sentence_to_vector(root.message, word2vec_model)
        agent.cosine_similarity = cosine_similarity_score(vector1, vector2)
    else:
        agent.cosine_similarity = 0
        
    # compute metrics for children
    for child in agent.children:
        compute_metrics(child, root, word2vec_model)

# print the agent tree with messages
def print_agent_tree(agent, level=0):
    indent = "  " * level
    print(f"{indent}Agent {agent.id}:")
    print(f"{indent}Message: {agent.message}")
    if agent.rouge_score is not None:
        print(f"{indent}Rouge Score: {agent.rouge_score:.4f}")
    if agent.cosine_similarity is not None:
        print(f"{indent}Cosine Similarity: {agent.cosine_similarity:.4f}")
    if agent.length_ratio is not None:
        print(f"{indent}Length Ratio: {agent.length_ratio:.4f}")
    print()
    
    for child in agent.children:
        print_agent_tree(child, level + 1)

# extract all scores from the tree for analysis
def extract_scores(agent, scores=None, depth=0):
    if scores is None:
        scores = {
            'agent_id': [],
            'depth': [],
            'rouge': [],
            'cosine': [],
            'length_ratio': []
        }
    
    scores['agent_id'].append(agent.id)
    scores['depth'].append(depth)
    scores['rouge'].append(agent.rouge_score)
    scores['cosine'].append(agent.cosine_similarity)
    scores['length_ratio'].append(agent.length_ratio)
    
    for child in agent.children:
        extract_scores(child, scores, depth + 1)
    
    return scores

# save results to json
def save_results(agent, filename="results.json"):
    def agent_to_dict(agent):
        agent_dict = {
            'id': agent.id,
            'message': agent.message,
            'rouge_score': agent.rouge_score,
            'cosine_similarity': agent.cosine_similarity,
            'length_ratio': agent.length_ratio,
            'children': [agent_to_dict(child) for child in agent.children]
        }
        return agent_dict
    
    result = agent_to_dict(agent)
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)

# load recipe data
def load_data(source="wikihow"):
    if source == "wikihow":
        try:
            wikihow = pd.read_json('wikihow.json', lines=True)
            wikihow = wikihow[['step_texts']]
            return wikihow
        except Exception as e:
            print(f"Error loading wikihow data: {e}")
            return None
    else:
        # fallback to sample data
        sample_data = {
            'step_texts': [
                {'text': "Place the ribbon spool on one side of the book and wrap it around. Place your ribbon spool just to the right or left of your wrapped book and then pull the loose end of the ribbon horizontally across the front center of the book. Stop when the loose end of the ribbon is a little bit past the edge of the book."},
                {'text': "Preheat your oven to 350°F. In a large bowl, mix flour, sugar, and salt. Add butter and mix until combined. Press the mixture into the bottom of a 9x13 inch baking pan. Bake for 15 minutes or until lightly golden."}
            ]
        }
        sample_df = pd.DataFrame(sample_data)
        return sample_df

# plot metrics
def plot_metrics(scores, filename="metrics_by_depth.png"):
    depths = sorted(set(scores['depth']))
    
    # average metrics by depth
    avg_rouge = []
    avg_cosine = []
    avg_length = []
    
    for d in depths:
        indices = [i for i, depth in enumerate(scores['depth']) if depth == d]
        avg_rouge.append(np.mean([scores['rouge'][i] for i in indices]))
        avg_cosine.append(np.mean([scores['cosine'][i] for i in indices]))
        avg_length.append(np.mean([scores['length_ratio'][i] for i in indices]))
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(depths, avg_rouge, 'o-', label='ROUGE Score')
    plt.xlabel('Depth')
    plt.ylabel('Average ROUGE')
    plt.title('ROUGE Score by Depth')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(depths, avg_cosine, 'o-', label='Cosine Similarity')
    plt.xlabel('Depth')
    plt.ylabel('Average Cosine Similarity')
    plt.title('Semantic Similarity by Depth')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(depths, avg_length, 'o-', label='Length Ratio')
    plt.xlabel('Depth')
    plt.ylabel('Average Length Ratio')
    plt.title('Message Length Ratio by Depth')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# main function
def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Run paraphrase propagation experiments")
    parser.add_argument("--model", type=str, default="gpt2", 
                      help="Model to use for paraphrasing: gpt2, distilgpt2, or a mistral model if Ollama is installed")
    args = parser.parse_args()
    
    print(f"Setting up LLM using model: {args.model}...")
    pipe = setup_llm(args.model)
    
    print("Loading word vectors...")
    word2vec_model = load_word_vectors()
    
    print("Loading data...")
    data = load_data()
    
    if data is None:
        print("Using default example message...")
        task = "Place the ribbon spool on one side of the book and wrap it around. Place your ribbon spool just to the right or left of your wrapped book and then pull the loose end of the ribbon horizontally across the front center of the book. Stop when the loose end of the ribbon is a little bit past the edge of the book."
    else:
        # randomly select a task
        indx = random.randint(0, data.shape[0]-1)
        task = data.iloc[indx]['step_texts'][0]['text']
    
    print("Original task:")
    print(task)
    print()
    
    # create results directory
    os.makedirs("results", exist_ok=True)
    
    # run multiple experiments with different tree structures
    num_experiments = 3
    max_agents_list = [10, 20, 30]
    
    for exp_idx, max_agents in enumerate(max_agents_list):
        print(f"Experiment {exp_idx+1}/{num_experiments} with {max_agents} agents")
        
        # build agent tree
        root_agent = build_agent_tree(max_agents)
        
        # visualize tree structure
        visualize_agent_tree(root_agent, f"results/tree_exp{exp_idx+1}")
        
        # propagate message
        propagate_task(root_agent, task, pipe)
        
        # compute metrics
        compute_metrics(root_agent, root_agent, word2vec_model)
        
        # visualize tree with rouge scores
        visualize_agent_tree_with_rouge(root_agent, f"results/tree_rouge_exp{exp_idx+1}")
        
        # extract scores for analysis
        scores = extract_scores(root_agent)
        
        # plot metrics
        plot_metrics(scores, f"results/metrics_exp{exp_idx+1}.png")
        
        # save results
        save_results(root_agent, f"results/results_exp{exp_idx+1}.json")
        
        # print tree (only for first experiment to avoid too much output)
        if exp_idx == 0:
            print_agent_tree(root_agent)
    
    print("Experiments completed!")

if __name__ == "__main__":
    main()
