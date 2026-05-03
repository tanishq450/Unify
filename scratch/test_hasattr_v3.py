
class LLM:
    def with_structured_output(self, schema):
        return "I have it"

class Wrapped:
    def __init__(self, llm):
        self._llm = llm
    def __getattr__(self, name):
        if name == "with_structured_output":
            raise AttributeError("Disabled")
        return getattr(self._llm, name)

llm = LLM()
wrapped = Wrapped(llm)

print(f"hasattr(wrapped, 'with_structured_output'): {hasattr(wrapped, 'with_structured_output')}")
try:
    wrapped.with_structured_output("schema")
except AttributeError as e:
    print(f"Caught: {e}")
