from prompt_manager import PromptManager
import re
from typing import Final


INJECTION_PATTERNS: Final[list[str]] = [
	r"ignore (your |all |previous )?instructions",
	r"system prompt.*disabled",
	r"new role",
	r"repeat.*system prompt",
	r"jailbreak",
]



def main():
    prompt_manager = PromptManager()
    print("-----------------------------------------")
    prompt_data = prompt_manager.load_prompt()
    print("Prompt data loaded successfully.")
    print("Prompt data:", prompt_data)
    print("-----------------------------------------")

if __name__ == "__main__":
    main()