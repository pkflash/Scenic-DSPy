import dspy
import os
import requests
from bs4 import BeautifulSoup
import html2text
from dotenv import load_dotenv

def scrapeSites(urls):
    if urls:
        context_passages = []  # Store all context in this array
        for url in urls:
            try:
                response = requests.get(url)
                response.raise_for_status() # Raise an exception for bad status codes
                soup = BeautifulSoup(response.text, 'html.parser') # Grab raw HTML text
                text_content = html2text.html2text(str(soup))  # Extract all HTML body Text and add to context
                context_passages.append(text_content)
            except requests.exceptions.RequestException as e:
                print(f"Error fetching URL {url}: {e}")
        
        # Return context array
        return context_passages
    else:
        # Return empty context array
        return []

if __name__ == "__main__":
    # Load key
    load_dotenv()
    key = os.getenv("API_KEY")

    # Compile Scenic Documentation URLs
    urls = [
        "https://docs.scenic-lang.org/en/latest/tutorials/fundamentals.html#",
        "https://docs.scenic-lang.org/en/latest/syntax_guide.html",
        "https://docs.scenic-lang.org/en/latest/tutorials/dynamics.html",
        "https://docs.scenic-lang.org/en/latest/tutorials/composition.html",
        "https://docs.scenic-lang.org/en/latest/language_reference.html",
        "https://docs.scenic-lang.org/en/latest/reference/general.html",
        "https://docs.scenic-lang.org/en/latest/reference/data.html",
        "https://docs.scenic-lang.org/en/latest/reference/region_types.html",
        "https://docs.scenic-lang.org/en/latest/reference/distributions.html",
        "https://docs.scenic-lang.org/en/latest/reference/statements.html",
        "https://docs.scenic-lang.org/en/latest/reference/classes.html",
        "https://docs.scenic-lang.org/en/latest/reference/sensors.html",
        "https://docs.scenic-lang.org/en/latest/reference/specifiers.html#",
        "https://docs.scenic-lang.org/en/latest/reference/operators.html",
        "https://docs.scenic-lang.org/en/latest/reference/functions.html",
        "https://docs.scenic-lang.org/en/latest/reference/visibility.html",
        "https://docs.scenic-lang.org/en/latest/reference/scene_generation.html",
        "https://docs.scenic-lang.org/en/latest/reference/dynamic_scenarios.html",
        "https://docs.scenic-lang.org/en/latest/options.html",
        "https://docs.scenic-lang.org/en/latest/api.html",

    ]
    # Initialize all necessary LM + rag objects
    lm = dspy.LM("openai/gpt-5.1", temperature=1.0, max_tokens=20000, api_key=key)
    dspy.configure(lm=lm)
    context = scrapeSites(urls)
    rag = dspy.ChainOfThought("context, question -> response")
    
    while True:
        query = input("Prompt the RAG model:")


        test_output = rag(context=context, question=query)
        print(test_output.response)

        # Directly write to script.scenic for generation requests
        if "Generate" in query:
            # Write to script
            with open("script.scenic", "w") as f:
                f.write(test_output.response)
    