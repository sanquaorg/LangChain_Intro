from langchain.chains.sequential import SimpleSequentialChain
import openai 
from langchain_openai import OpenAI 
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentType, initialize_agent, load_tools,Agent,tool
from langchain.memory import ConversationBufferMemory


def simple_llm_prompt():
  prompt=PromptTemplate.from_template("What is the capital of  {place}?")
  llm = OpenAI(temperature=0.3)
  chain=LLMChain(llm=llm,prompt=prompt)
  output = chain.run("Punjab")
  print(output)
# my_secret = os.environ['SECRET_KEY']


# ---------------------------------------------------------
def simple_seq():
# Simple Sequential chain


# LLM to get e commerce store name from product name
  prompt = PromptTemplate.from_template("What is the name of the e-commerce store where I can find {product_name}?")
  llm = OpenAI(temperature=0.3)
  chain = LLMChain(llm=llm, prompt=prompt)
  
  # LLM to get products name from e commerce store name
  ecommerce_prompt = PromptTemplate.from_template("What are the popular products available at {store_name}?")
  ecommerce_llm = OpenAI(temperature=0.3)
  ecommerce_chain = LLMChain(llm=ecommerce_llm, prompt=ecommerce_prompt)
  
  #Overall chain
  chain = SimpleSequentialChain(chains=[chain, ecommerce_chain], verbose=True)
  
  output=chain.run("phones")
  print(output)
# ---------------------------------------------------------

#Sequential Chain - Refer Lanchain Docs

# https://python.langchain.com/docs/modules/chains/sequential/
# ---------------------------------------------------------
#Agents in Langchain

llm=OpenAI(temperature=.7)
tools=load_tools(["wikipedia","llm-math"],llm=llm)
agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
# output=agent.run("Holw Old is Varun Dhawan in 2023?")
# print(output)
# ---------------------------------------------------------
#Memory in Langchain
llm=OpenAI(temperature=.3)
prompt=PromptTemplate.from_template("What is the name of the e-commerce store that sells {product}?")
chain=LLMChain(llm=llm,prompt=prompt,memory=ConversationBufferMemory())
output=chain.run("fruits")
output=chain.run("books")
output=chain.run("tech gadgets")
output=chain.run("PC parts")
print(chain.memory.buffer)
print(output)


  