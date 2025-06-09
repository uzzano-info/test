"""Simple LangChain sample that asks a question via OpenAI."""

import argparse
import os

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def main() -> None:
    """Run a single question through the chat model."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY environment variable is not set."
        )

    parser = argparse.ArgumentParser(description="Minimal LangChain example")
    parser.add_argument(
        "question",
        nargs="?",
        default="LangChain 시작을 도와줘",
        help="Question to ask the model",
    )
    parser.add_argument(
        "--model",
        default="GPT-4.1 mini",
        help="OpenAI model name",
    )
    args = parser.parse_args()

    llm = ChatOpenAI(model=args.model)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"question": args.question})
    print(result.content)

if __name__ == "__main__":
    main()
