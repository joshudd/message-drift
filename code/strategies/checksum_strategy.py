def checksum_propagation(root, task, pipe):
    """propagate task with checksum to detect and correct information loss"""
    # extract key terms from original message
    keywords = [word for word in task.lower().split() if len(word) > 4][:5]
    checksum = "-".join(keywords)
    
    # append checksum to original task
    root.message = f"{task} [CHECKSUM: {checksum}]"
    
    def propagate(agent):
        print(f"agent {agent.id} processing message with checksum strategy")
        if not agent.children:
            return
            
        for child in agent.children:
            if agent.message:
                # extract checksum from parent message
                parent_message = agent.message
                parent_checksum = None
                if "[CHECKSUM:" in parent_message:
                    parts = parent_message.split("[CHECKSUM:")
                    parent_message = parts[0].strip()
                    parent_checksum = parts[1].split("]")[0].strip()
                
                # generate paraphrase
                prompt = f"""<s>[INST] paraphrase this instruction but preserve all important details:

{parent_message}

[/INST]"""
                try:
                    response = pipe(prompt)[0]["generated_text"]
                    if response.startswith("answer:"):
                        response = response[7:].strip()
                    
                    # reattach checksum
                    if parent_checksum:
                        child.message = f"{response} [CHECKSUM: {parent_checksum}]"
                    else:
                        child.message = response
                except Exception as e:
                    print(f"error in paraphrase: {e}")
                    child.message = parent_message
            
            propagate(child)
    
    propagate(root)
    return root 