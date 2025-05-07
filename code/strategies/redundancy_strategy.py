def redundancy_propagation(root, task, pipe):
    """propagate task using redundancy to maintain message integrity"""
    # identify key information in task
    sentences = task.split(". ")
    
    # create redundant message by repeating important points
    important_parts = sentences if len(sentences) <= 2 else sentences[:2]
    redundant_parts = "IMPORTANT: " + ". ".join(important_parts) + "."
    
    # create message with redundancy
    root.message = f"{task}\n\n{redundant_parts}"
    
    def propagate(agent):
        print(f"agent {agent.id} processing message with redundancy strategy")
        if not agent.children:
            return
            
        for child in agent.children:
            if agent.message:
                # extract original and redundant parts
                message_parts = agent.message.split("\n\n")
                main_message = message_parts[0]
                redundant_info = message_parts[1] if len(message_parts) > 1 else ""
                
                # paraphrase only main part
                prompt = f"""<s>[INST] reword this instruction in your own words:

{main_message}

[/INST]"""
                try:
                    response = pipe(prompt)[0]["generated_text"]
                    if response.startswith("answer:"):
                        response = response[7:].strip()
                    
                    # reattach redundant part
                    if redundant_info:
                        child.message = f"{response}\n\n{redundant_info}"
                    else:
                        child.message = response
                except Exception as e:
                    print(f"error in paraphrase: {e}")
                    child.message = agent.message
            
            propagate(child)
    
    propagate(root)
    return root 