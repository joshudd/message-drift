import random
import re
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def needle_in_haystack_propagation(root, task, pipe):
    """propagate task with embedded unique identifiers to track information preservation"""
    
    # create key-value pairs that will serve as our "needles"
    task_words = [word for word in task.lower().split() if len(word) > 3]
    selected_words = random.sample(task_words, min(5, len(task_words)))
    
    # create unique identifiers for each needle
    needles = {}
    for i, word in enumerate(selected_words):
        needle_id = f"nid{i}"
        needles[needle_id] = word
    
    # embed needles in the task
    modified_task = task
    for needle_id, word in needles.items():
        # insert the needle ID after mentions of the word
        pattern = f"\\b{re.escape(word)}\\b"
        modified_task = re.sub(pattern, f"{word} [{needle_id}]", modified_task, count=1, flags=re.IGNORECASE)
    
    # provide a legend at the end of the task
    legend = "\n\nLegend (keep these identifiers intact):"
    for needle_id, word in needles.items():
        legend += f"\n[{needle_id}] marks '{word}'"
    
    root.message = f"{modified_task}{legend}"
    
    def propagate(agent):
        print(f"agent {agent.id} processing message with needle-in-haystack strategy")
        if not agent.children:
            return
        
        # process in smaller batches to conserve memory
        batch_size = 3
        for i in range(0, len(agent.children), batch_size):
            batch = agent.children[i:i+batch_size]
            
            for child in batch:
                if agent.message:
                    # extract legend and main message
                    message_parts = agent.message.split("\n\nLegend")
                    main_message = message_parts[0]
                    legend_part = "\n\nLegend" + message_parts[1] if len(message_parts) > 1 else ""
                    
                    # paraphrase while preserving needle tags
                    prompt = f"""<s>[INST] paraphrase this instruction in your own words, but keep all [nidX] tags exactly as they appear:

{main_message}

[/INST]"""
                    try:
                        response = pipe(prompt)[0]["generated_text"]
                        if response.startswith("answer:"):
                            response = response[7:].strip()
                        
                        # reattach legend
                        child.message = f"{response}{legend_part}"
                        
                        # verify needles were preserved, recover any lost ones
                        for needle_id in needles.keys():
                            if f"[{needle_id}]" not in child.message and legend_part:
                                # needle was lost, add warning and recovery hint
                                child.message = f"{child.message}\n\nWARNING: Some tracking identifiers were lost in transmission. Please preserve all [nidX] tags."
                    except Exception as e:
                        print(f"error in paraphrase: {e}")
                        child.message = agent.message
            
            # clear cuda cache after each batch if using gpu
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        
        # recursively process children
        for child in agent.children:
            propagate(child)
    
    propagate(root)
    
    # store needle preservation metrics
    def count_preserved_needles(agent):
        if not agent.message:
            return 0
        
        # extract main message (without legend)
        message_parts = agent.message.split("\n\nLegend")
        main_message = message_parts[0]
        
        count = 0
        for needle_id in needles.keys():
            # only count if needle ID is in the main message content, not just in the legend
            if f"[{needle_id}]" in main_message:
                count += 1
        return count
    
    # traverse tree and assess needle preservation
    total_agents = 0
    total_needles_preserved = 0
    
    def traverse_and_count(agent):
        nonlocal total_agents, total_needles_preserved
        
        total_agents += 1
        preserved = count_preserved_needles(agent)
        total_needles_preserved += preserved
        
        # store needle preservation rate as an attribute
        agent.needle_preservation = preserved / len(needles) if needles else 1.0
        # print("HERE", needles, agent.needle_preservation)
        
        for child in agent.children:
            traverse_and_count(child)
    
    traverse_and_count(root)
    
    # find agents furthest from root
    def get_tree_depth_and_leaf_nodes(agent, depth=0):
        if not agent.children:
            return depth, [agent]
        
        max_depth = depth
        furthest_agents = []
        
        for child in agent.children:
            child_depth, child_furthest = get_tree_depth_and_leaf_nodes(child, depth+1)
            if child_depth > max_depth:
                max_depth = child_depth
                furthest_agents = child_furthest
            elif child_depth == max_depth:
                furthest_agents.extend(child_furthest)
        
        return max_depth, furthest_agents
    
    max_tree_depth, furthest_agents = get_tree_depth_and_leaf_nodes(root)
    
    # count actual preserved needles by type
    needle_preservation_by_id = {needle_id: 0 for needle_id in needles.keys()}
    total_agents_with_message = 0
    
    def analyze_needle_preservation(agent):
        nonlocal total_agents_with_message
        
        if not agent.message:
            return
            
        total_agents_with_message += 1
        message_parts = agent.message.split("\n\nLegend")
        main_message = message_parts[0]
        
        for needle_id in needles.keys():
            if f"[{needle_id}]" in main_message:
                needle_preservation_by_id[needle_id] += 1
        
        for child in agent.children:
            analyze_needle_preservation(child)
    
    analyze_needle_preservation(root)
    
    print(f"\n{'='*50}")
    print(f"NEEDLE PRESERVATION ANALYSIS:")
    
    for needle_id, word in needles.items():
        preservation_rate = needle_preservation_by_id[needle_id] / total_agents_with_message if total_agents_with_message else 0
        print(f"Needle '{needle_id}' for word '{word}': preserved in {needle_preservation_by_id[needle_id]}/{total_agents_with_message} agents ({preservation_rate:.2%})")
    
    overall_needle_preservation = sum(needle_preservation_by_id.values()) / (total_agents_with_message * len(needles)) if total_agents_with_message and needles else 0
    print(f"Overall needle preservation rate: {overall_needle_preservation:.2%}")
    print(f"{'='*50}\n")
    
    print(f"\n{'='*50}")
    print(f"FURTHEST AGENTS (depth: {max_tree_depth}):")
    for agent in furthest_agents:
        # extract main message (without legend)
        if agent.message:
            message_parts = agent.message.split("\n\nLegend")
            main_message = message_parts[0]
            legend_part = "\n\nLegend" + message_parts[1] if len(message_parts) > 1 else ""
            
            # count needles in main message
            preserved_in_main = 0
            for needle_id in needles.keys():
                if f"[{needle_id}]" in main_message:
                    preserved_in_main += 1
            
            print(f"\nAgent ID: {agent.id}")
            print(f"Needle preservation rate: {agent.needle_preservation:.2f}")
            print(f"Needles in main message: {preserved_in_main}/{len(needles)}")
            print(f"Message:\n{agent.message}")
        else:
            print(f"\nAgent ID: {agent.id}")
            print(f"No message")
    print(f"{'='*50}\n")
    
    # calculate preservation metrics
    avg_preservation = total_needles_preserved / (total_agents * len(needles)) if needles and total_agents else 0
    print(f"average needle preservation rate: {avg_preservation:.2f}")
    
    return root

def plot_needle_preservation(root, filename="needle_preservation.png"):
    """plot needle preservation rates across the tree"""
    # create a graph
    G = nx.Graph()
    
    # add nodes with needle preservation rates
    def add_nodes(agent, parent_id=None):
        # skip if agent doesn't have needle_preservation attribute
        if not hasattr(agent, "needle_preservation"):
            return
            
        G.add_node(agent.id, preservation=agent.needle_preservation)
        if parent_id is not None:
            G.add_edge(parent_id, agent.id)
        for child in agent.children:
            add_nodes(child, agent.id)
    
    add_nodes(root)
    
    # check if there are nodes to plot
    if not G.nodes():
        print("no needle preservation data to plot")
        return
    
    # layout
    pos = nx.spring_layout(G, seed=42)  # added seed for reproducibility
    
    # get preservation rates
    preservation_rates = nx.get_node_attributes(G, 'preservation')
    
    # print actual values for debugging
    print("node preservation rates:")
    for node, rate in preservation_rates.items():
        print(f"node {node}: {rate:.2f}")
    
    # create colormap - ensure we're getting the actual values
    node_colors = [preservation_rates.get(node, 0) for node in G.nodes()]
    
    # create figure with proper axes setup for colorbar
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # draw network with explicit normalization
    nodes = nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          cmap=plt.cm.RdYlGn,
                          vmin=0, vmax=1,
                          node_size=500)
    
    # draw edges and labels separately
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # create a scalar mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    
    # add colorbar to the figure
    cbar = fig.colorbar(sm, ax=ax, label='needle preservation rate')
    
    plt.title('needle preservation across agent tree')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() 