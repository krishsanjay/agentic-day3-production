import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

class PromptManager:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)

    def load_prompt(self, version: str = "current") -> Dict[str, Any]:
        """
        Load the prompt configuration from a YAML file.
        """
        if version == "current":
            prompt_file = self.prompts_dir / "support_agent_v1.yaml"
        else:
            prompt_file = self.prompts_dir / f"support_agent_{version}.yaml"    

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r') as f:
            prompt_data = yaml.safe_load(f)
        
        return prompt_data
    
    def compile_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Compile the prompt data into a single string.
        """
        system_prompt = prompt_data.get("system_prompt", "")
        user_instructions = prompt_data.get("user_instructions", "")
        example_interactions = prompt_data.get("example_interactions", [])
        
        compiled_prompt = system_prompt + "\n\n" + user_instructions + "\n\n"
        
        for interaction in example_interactions:
            compiled_prompt += f"User: {interaction['user']}\nAgent: {interaction['agent']}\n\n"
        
        return compiled_prompt