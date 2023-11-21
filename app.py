import os
from dotenv import load_dotenv

# from langchain import PromptTemplate
# from langchain.agents import initialize_agent, Tool
# from langchain.agents import AgentType
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import MessagesPlaceholder
# from langchain.memory import ConversationSummaryBufferMemory
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.summarize import load_summarize_chain
# from langchain.tools import BaseTool
# from pydantic import BaseModel, Field
# from typing import Type
from bs4 import BeautifulSoup
import requests
import json
# from langchain.schema import SystemMessage
# from fastapi import FastAPI

load_dotenv()

browerless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
    })

    headers = {
        'x-api-key': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

# search("What public deals has Nike done?")


# 2. Tool for scraping
def scrape_website( objective: str, url: str):
    print(f"Scraping {url} for {objective}")
    
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json'
    }

    data = {
        "url": url,
    }

    data_json = json.dumps(data)

    post_url = f"https://chrome.browserless.io/content?token={browerless_api_key}"

    response = requests.post(post_url, headers=headers, data=data_json)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()

        print(f"Content: {text}")

        return text
    else:
        print(f"Error: {response.status_code}")
        return None
    
scrape_website("What deals did nike do?", "https://www.hotnewhiphop.com/646142-nikes-largest-athlete-endorsement-deals")

# 3. Create langchain agent with tools above