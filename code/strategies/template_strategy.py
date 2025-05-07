def template_propagation(root, task, pipe):
    """propagate task using structured templates to preserve key information"""
    # parse task into structured template
    sentences = task.split(". ")
    
    # create template
    steps = []
    for i, sentence in enumerate(sentences):
        if sentence:
            steps.append(f"Step {i+1}: {sentence}")
    
    template = "\n".join(steps)
    root.message = template
    
    def propagate(agent):
        print(f"agent {agent.id} processing message with template strategy")
        if not agent.children:
            return
            
        for child in agent.children:
            if agent.message:
                template_lines = agent.message.split("\n")
                structured_steps = []
                
                # paraphrase each step individually
                for step in template_lines:
                    if step.startswith("Step "):
                        step_num = step.split(":")[0]
                        step_content = ":".join(step.split(":")[1:]).strip()
                        
                        prompt = f"""<s>[INST] reword this instruction keeping all key details:

{step_content}

[/INST]"""
                        try:
                            response = pipe(prompt)[0]["generated_text"]
                            if response.startswith("answer:"):
                                response = response[7:].strip()
                            
                            structured_steps.append(f"{step_num}: {response}")
                        except Exception as e:
                            print(f"error paraphrasing step: {e}")
                            structured_steps.append(step)
                
                child.message = "\n".join(structured_steps)
            
            propagate(child)
    
    propagate(root)
    return root 