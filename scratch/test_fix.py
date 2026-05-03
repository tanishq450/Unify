

class _MarkdownStrippingLLM:
    def __init__(self, llm):
        self._llm = llm
    def __getattr__(self, name):
        if name == "with_structured_output":
             raise AttributeError("with_structured_output is disabled on this wrapper to ensure output cleaning.")
        return getattr(self._llm, name)

class MockLLM:
    def with_structured_output(self, schema):
        return "Real Structured Output"

def mock_transformer_logic(llm):
    # This mimics how LangChain components often check for the method
    method = getattr(llm, "with_structured_output", None)
    if method:
        print("Using structured output...")
        return method("schema")
    else:
        print("Falling back to prompt-based extraction...")
        return "Prompt Result"

real_llm = MockLLM()
wrapped_llm = _MarkdownStrippingLLM(real_llm)

print("Test 1: Wrapped LLM")
result = mock_transformer_logic(wrapped_llm)
print(f"Result: {result}")

print("\nTest 2: Real LLM")
result = mock_transformer_logic(real_llm)
print(f"Result: {result}")
