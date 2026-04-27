import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Initialize the client with your API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Use the generate method
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Say hello"
)
print(response.text)