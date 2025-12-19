"""
A simple translation agent example for testing and demonstration.
This file provides the create_translation_agent function used in e2e tests.
"""

from core.compiler import compile_agent
from encompass import branchpoint, record_score

def create_translation_agent(llm=None):
    """
    Creates a translation agent that generates code based on choices.
    """
    @compile_agent
    def translation_agent():
        # Choice 1: Signature Style
        sig_style = branchpoint("signature_style", options=[0, 1, 2])
        
        # Choice 2: Body Style
        body_style = branchpoint("body_style", options=[0, 1])
        
        # Mock result generation
        if sig_style == 0 and body_style == 0:
            result = "template <typename T>\nT add(T a, T b) {\n    return a + b;\n}"
            score = 150.0
        else:
            result = "int add(int a, int b) { return a + b; }"
            score = 50.0
            
        record_score(score)
        return result
        
    return translation_agent

if __name__ == "__main__":
    # Example usage
    agent = create_translation_agent()
    print("Agent created successfully.")
