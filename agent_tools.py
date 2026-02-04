from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import os
import requests
load_dotenv()

@tool
def get_news(city: str) -> str:
    """Get the latest news regarding a city"""
    api_key=os.getenv("NEWS_API_KEY")

    url = "https://newsapi.org/v2/everything"
    params={
       "q":city,
       "apiKey":api_key,
       "pageSize":1,
       "sortBy":"publishedAt"
    }
    r = requests.get(url, params=params)
    data = r.json()

    if "articles" not in data:
        return "No news found."

    news = []
    for a in data["articles"]:
        news.append(f"- {a['title']} ({a['source']['name']})")

    return "\n".join(news)

@tool

def get_facts(city: str) -> str:
    """Get interesting factual information about a city from Wikipedia"""

    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{city}"

    r = requests.get(url, headers={"User-Agent": "LangChainAgent/1.0"})
    
    if r.status_code != 200:
        return "No factual information found."

    data = r.json()
    return data.get("extract", "No facts available.")



prompt="you are provided with the different tools one is to get the news regarding that city and other is to get some interesting facts about that city . use the tools when ever necessary to get the information need to answer the user query"
    
model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro",
    temperature=0.7,
    max_tokens=1000
)


agent = create_agent(
    model=model,
    tools=[get_news, get_facts],
    system_prompt=prompt
)

response = agent.invoke(
    {"messages": [("human", "What is the news in Belagavi?")]}
)

print(response["messages"][-1].content)
