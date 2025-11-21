import dspy
import os
import requests
from bs4 import BeautifulSoup
import html2text
from dotenv import load_dotenv

# Load key and teacher/student LMs 
load_dotenv()
key = os.getenv("API_KEY")
teacherLM = dspy.LM("openai/gpt-5.1", api_key=key)
studentLM = dspy.LM("openai/gpt-4o-mini", api_key=key)

