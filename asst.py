from langchain.chains import LLMChain,SequentialChain,ConversationChain
from langchain.prompts import ChatPromptTemplate
from langchain_google_gen_ai import ChatGoogleGenerativeAI
from langchain.output_parser import ResponseSchema
from langchain.output_parser import StructuredOutputParser
from landchain.memory import ConversationBufferWindowMemory
from langchain.chains.llm_router import LLMRouterChain
from langchain.chains.router import MultiPromptChain

llm=ChatGoogleGenAI(
    model="models/gemini-2.5-pro",
    temperature=0.7,
    max_tokens=100
)

memory=ConversationBufferWindowMemory(k=1)

conversation=converastionChain(
    llm=llm,
    memory=memory,
    verbose=True
)

error_detect_prompt=ChatPromptTemplate(
    input_variables=["code"],
    template="""you are misra-c violations code debugger.
    need to identify the violations commited in the code
    1. Analyse the given c code properly
    2. find out the potential violations in the code
    3. give the rule numbers for the violated rules with description


    output should be in json format 
    {
       "error":"error",
       "rule": "rule violated"
       "descriptiom": "description of the rule violated"
    }


    c code:
    {code}
    """
)

error_detect__chain=LLMChain(
    llm=llm,
    prompt=error_detect_prompt,
    output_key="error_report"
)

error_fix_prompt= ChatPromptTemplate(
    input_variables=["code","error_report"],
    template=""" You are expert in fixing the misra-c violations.
    you are provided with the code and error_report that contain rules violated and descripton.
    you should,
    1.Analyse the code given and error_report properly.
    2.Then fix the violations using the best practises.
    3.Then give the corrected code only in the JSON format


    c code:
    {code}

    error_report:
    {error_report}
    
    """
)

error_fix_chain=LLMChain(
    llm=llm,
    prompt=error_fix_prompt,
    output_key="final_output"
)


final_chain=SequentialChain(
    chains=["error_detect_chain","error_fix_chain"],
    input_variables=["code"],
    output_variables=["final_output","error_report"],
    verbose=True
)
