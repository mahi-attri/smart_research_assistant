import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Basic settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
APP_NAME = "Smart Research Assistant"
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"