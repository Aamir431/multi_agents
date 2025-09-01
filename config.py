# Load environment variables from a .env file if it exists.
from dotenv import load_dotenv
import os

load_dotenv() # This loads the variables from the .env file

# --- OpenAI API Configuration ---
# ONLY use the environment variable. Delete the hardcoded key.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Get the key from the .env file

# Check if the key was actually found
if not OPENAI_API_KEY:
    raise ValueError("ERROR: OpenAI API key not found. Please set it in the .env file.")

# The model to use for generating reports and other tasks.
OPENAI_MODEL = "gpt-4"

# --- File and Directory Paths ---
REPORTS_DIR = "reports"
DATA_DIR = "data"

# --- Other Configurations ---
AI_TEMPERATURE = 0.7
VERBOSE_LOGGING = True