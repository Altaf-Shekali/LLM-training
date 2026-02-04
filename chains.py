from langchain.chains import LLMChain,SequentialChain
from langchain_google_gen_ai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

llm=ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro",
    temperature=0.7
)


error_detection_prompt=PromptTemplate(
    input_variables=["code"],
    template="""you are a expert in pointing out the Misra-C violtions written in the c code 
    so carefully examine the code and give the error report where the errors are and how to fix them

    rules to be followed:
    analyse the code properly for misra-c violations
    give the errors in format

    source_code:
    {code}"""
)

error_detect_chain=LLMChain(
    llm=llm,
    prompt=error_detection_prompt,
    output_key="error_report"
)

error_fix_prompt=PromptTemplate(
    input_variables=["code","error_report"],
    template=""" you are an expert misra-c violation reviewer and fixer.
    you are given with error report and the python code analyse the code for misra-c violations and fix the with best practises
    corrected ouput should be in the JSON format only """
)

error_fix_chain=LLMChain(
    llm=llm,
    prompt=error_fix_prompt,
    output_key="final_answer"
)


final_chain=SequentialChain(
    chains=[error_detect_chain,error_fix_chain],
    input_variables=["code"],
    output_variables=["error_report","final_answer"],
    verbose=True
)


