try:
    from google import genai
    print("google-genai is installed")
except ImportError:
    print("google-genai is NOT installed")

try:
    import google.generativeai as gai
    print("google-generativeai is installed")
except ImportError:
    print("google-generativeai is NOT installed")
