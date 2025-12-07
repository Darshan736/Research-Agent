from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from tools import search_tool, wiki_tool, save_tool

load_dotenv() 


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text
            {format_instructions}
            """,
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

user_query = input("What can I help you research? ")


raw_response = agent_executor.invoke({"input": user_query})

try:
    raw_output = raw_response["output"]

    # 1. Normalize to a single string
    if isinstance(raw_output, list):
        parts = []
        for chunk in raw_output:
            if isinstance(chunk, dict) and "text" in chunk:
                parts.append(chunk["text"])
            else:
                parts.append(str(chunk))
        text = "".join(parts)
    else:
        text = str(raw_output)

    stripped = text.strip()
    if stripped.startswith("```"):
        
        stripped = stripped.split("\n", 1)[1]
       
        if "```" in stripped:
            stripped = stripped.rsplit("```", 1)[0]

    structured_response = parser.parse(stripped)
    print(structured_response)

except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response:", raw_response)

