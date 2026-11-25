from google import genai
import os
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]


#export GEMINI_API_KEY="api_key"


client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Explain how AI works in a few words"
)


print(interaction.output_text)

