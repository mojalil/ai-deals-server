import os
from dotenv import load_dotenv

from langchain import PromptTemplate
# from langchain.agents import initialize_agent, Tool
# from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
# from langchain.prompts import MessagesPlaceholder
# from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
# from langchain.schema import SystemMessage
# from fastapi import FastAPI

load_dotenv()

browerless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

CHUNK_SIZE = 10000

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

        if len(text) > CHUNK_SIZE:
            output = summary(objective, text)
            return output
        else:    
            return text
    else:
        print(f"Error: {response.status_code}")
        return None
    
# scrape_website("What deals did nike do?", "https://www.hotnewhiphop.com/646142-nikes-largest-athlete-endorsement-deals")

def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=CHUNK_SIZE, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    # Input for scrape_website
    objective: str = Field(description="The objective and the task that the users give to the agent")
    url: str = Field(description="The url of the website to scrape")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")
# 3. Create langchain agent with tools above