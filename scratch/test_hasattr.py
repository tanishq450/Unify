
class A:
    def foo(self):
        return 1

class B:
    def __init__(self, obj):
        self.obj = obj
    def __getattr__(self, name):
        if name == "foo":
            raise AttributeError("Disabled")
        return getattr(self.obj, name)

a = A()
b = B(a)

print(f"hasattr(b, 'foo'): {hasattr(b, 'foo')}")
try:
    b.foo()
except AttributeError as e:
    print(f"Caught: {e}")
