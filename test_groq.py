import os
from langchain_groq import ChatGroq

def test():
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)
    res = llm.invoke("hello")
    print(res)

test()
