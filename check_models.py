import google.generativeai as genai
import os

# ✅ Use API key directly (Not recommended for security reasons)
api_key = "AIzaSyCYSjBMP_fK91_6UMwvnSg6yN-2Vd2gyEg"

# ✅ Print API key (partially) for debugging
if api_key:
    print(f"🔑 API Key Loaded: {api_key[:5]}********")
else:
    print("❌ Oops! API Key is missing. Check your .env file!")
    exit(1)

# ✅ Configure Google Generative AI
genai.configure(api_key=api_key)

# ✅ List available models
try:
    models = genai.list_models()
    print("\n✅ Available Models:")
    for model in models:
        print(f"- {model.name}")
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
