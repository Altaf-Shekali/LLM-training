from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain_google_gen_ai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


llm=ChatGoogleGenerativeAI(
     
    model="models/gemini-2.5-pro",
    temperature=0.3,
    max_tokens=500
)

error_schemas = [
    ResponseSchema(name="error", description="MISRA-C violation name"),
    ResponseSchema(name="rule", description="MISRA-C rule number"),
    ResponseSchema(name="description", description="Explanation of the violation")
]

error_parser = StructuredOutputParser.from_response_schemas(error_schemas)
format_instructions = error_parser.get_format_instructions()




error_detect_prompt=ChatPromptTemplate.from_template(
        """you are misra-c violations code debugger.
    need to identify the violations commited in the code
    1. Analyse the given c code properly
    2. find out the potential violations in the code
    3. give the rule numbers for the violated rules with description


    {format_instructions}


    c code:
    {code}
    """
)

error_detect_chain=LLMChain(
    llm=llm,
    prompt=error_detect_prompt,
    output_key="error_report"
)


error_fix_prompt= ChatPromptTemplate.from_template(
    """ You are expert in fixing the misra-c violations.
    you are provided with the code and error_report that contain rules violated and descripton.
    you should,
    1.Analyse the code given and error_report properly.
    2.Then fix the violations using the best practises.
    3.Then give the corrected code only in the JSON format

    JSON format:
{
    "corrected_code": "<fixed C code>"
}
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
    chains=[error_detect_chain,error_fix_chain],
    input_variables=["code"],
    output_variables=["final_output","error_report"],
    verbose=True
)

result = final_chain.invoke({
    "code": """
    int add(int a, int b) {
        return a + b;
    }
    """,
    "format_instructions": format_instructions
})

print(result["error_report"])
print(result["final_output"])

