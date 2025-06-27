from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool
from dotenv import load_dotenv
import os  
import requests
import chainlit as cl

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# function to get coin price
@function_tool
def get_cryto_price(currency):
    
    "Get the current price of a cryptocurrency."

    url = f"https://api.binance.com/api/v3/ticker/price?symbol={currency}"
    response = requests.get(url)
    data = response.json()
    return f"The current price of {currency} is {data['price']} USD."

# OpenAI Client 
external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",

)

# OpenAI Model
model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client
)

# create a RunConfig 
config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)


# create an Agent with function_tool

crypto_agent = Agent(
    name = "Crypto Coins Agent",
    instructions = """You are a Crypto Price helpful Assistant. Provide current real-time cryptocurrency prices,
     market data (24h change, market cap, volume), and key details (ATH, supply, links) in the
     user's preferred currency. Source data from CoinGecko/CoinMarketCap. Be concise, accurate,
     and mention if data is delayed. Example: 'Bitcoin is at $63,200 (+2.1%), $1.2T market cap.""",
    tools = [get_cryto_price],
)

# Chainlit setup

@cl.on_chat_start

async def start_massage():
    await cl.Message(content="""
                     💰 Welcome to the world of Crypto Currency! 🚀 Let's get started..
                     """).send()

#----------------

@cl.on_message

async def my_message(msg: cl.Message):
    user_input = msg.content 

    response = Runner.run_sync(
        crypto_agent,
        user_input,
        run_config = config
    )
    await cl.Message(
        content = response.final_output
    ).send()