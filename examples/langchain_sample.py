# LangChain minimal example
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# You must set the OPENAI_API_KEY environment variable before running this.

def main():
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])
    chain = prompt | llm
    result = chain.invoke({"question": "LangChain 시작을 도와줘"})
    print(result.content)

if __name__ == "__main__":
    main()
