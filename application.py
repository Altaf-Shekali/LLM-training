from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_gen_ai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain

llm=ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro",
    temperature=0.3,
    max_tokens=500
)

memory=ConversationBufferWindowMemory(k=2)

conversation_chain=ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

schemas = [
    ResponseSchema(name="error", description="MISRA-C violation"),
    ResponseSchema(name="rule", description="MISRA-C rule number"),
    ResponseSchema(name="description", description="Explanation")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()


coding_prompt=ChatPromptTemplate.from_template(
    """
    You are a MISRA-C compliance checker.

    {format_instructions}

    Analyze the following C code:
    {input}
    """
    )

coding_chain=LLMChain(
    llm=llm,
    prompt=coding_prompt,
    output_key="error_report"
)

router_prompt = ChatPromptTemplate.from_template(
    """
    You are a router.

    Decide which destination is best for the user input.

    Destinations:
    - conversation: general discussion or questions
    - misra: C code analysis and MISRA checking

    Return ONLY the destination name.
    Respond in JSON:
     {
    "destination": "<destination>",
    "next_inputs": {
    "input": "<original user input>"
    }
    }

    User input:
    {input}
    """
    )

router_chain = LLMRouterChain.from_llm(
    llm=llm,
    prompt=router_prompt
)

destination_chains = {
    "conversation": conversation_chain,
    "misra": coding_chain
}


default_chain = conversation_chain


app = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

app.invoke({"input": "Explain embeddings in simple words"})







