from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import operator
from langchain.messages import AnyMessage,SystemMessage,HumanMessage
from typing_extensions import TypedDict,Annotated
from typing import Literal
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


@tool

def multiply(a:int,b:int)->int:
    """ perform muliplication on both numbers and return product"""

    return a*b

@tool
def addition(a:int,b:int)->int:
    """prform addition on both the numbers and return the sum"""

    return a+b

llm=ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro",
    temperature=0.3,
    max_tokens=500
)

tools=[addition,multiply]
tools_by_name={tool.name: tool for tool in tools}
model_with_tools=model.bind_tools(tools)


class MessageState(TypedDict):
     messages=Annotated[list[AnyMessage],operator.add]
     llm_calls:int


def llm_call(State:dict):
    """llm to decide whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state:dict):
    """perform the tool call"""
    result=[]
    for tool_call in state["messages"][-1].tool_calls:
        tool= tools_by_name[tool_call["name"]]
        observation= tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation,tool_call_id=tool_call["id"]))

    return {"messages":result}


def should_continue(state:MessageState)->Literal["tool_node",END]:
     """ decide if we should stop the loop or stop based upon the whether tool node made call or not"""

     messages=state["messages"]
     last_message= messages[-1]


     if last_message.tool_calls:
            
         return END
   



agent_builder=StateGraph(MessageState)

agent_builder.add_node("llm_call", lla_call)
agent_builder.add_node("tool_node", tool_call)

agent_builder.add_edge(STATE, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_call",END]
)
agent_builder.add_edge("tool_node","llm_call")

agent_builder.compile()

display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()