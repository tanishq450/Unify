
class B:
    def foo(self, *args, **kwargs):
        raise AttributeError("Disabled")

b = B()
print(f"hasattr(b, 'foo'): {hasattr(b, 'foo')}")
try:
    b.foo()
except AttributeError as e:
    print(f"Caught: {e}")
