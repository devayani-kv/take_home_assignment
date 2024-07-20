import os
os.environ["OPENAI_API_KEY"] = "sk-proj-0GT7K5HQiFRZ42GQHo8rT3BlbkFJotR609bAI4DKyLatRsGw"
os.environ["TAVILY_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBENeAhSmK2pdyJds1YGj0XyLZcl2k5ATc"

import numpy as np
import pandas as pd
import operator 
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List
from typing_extensions import TypedDict
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph

import re
import g4f
import asyncio

_providers = [provider.__name__
    for provider in g4f.Provider.__providers__
    if provider.working]

async def run_provider(provider: g4f.Provider.BaseProvider, p):
    try:
        print('PROVIDER = ########################################################################### ',provider)
        response = await g4f.ChatCompletion.create_async(
            model=g4f.models.default,
            messages=[{ "role": "system", "content": "You are a friendly chatbot who always responds as superhuman intelligence AI."},
    { "role": "user", "content": p }],
            provider=provider,
        )
        return response
    except Exception as e:
        print(e)

async def run_all(prompt):
    """Runs all providers concurrently and returns a list of their responses.

    Returns:
        A list of g4f.ChatCompletion objects, each containing the response from a provider.
    """
    p = prompt
    calls = [run_provider(provider, p) for provider in _providers]
    responses = await asyncio.gather(*calls)
    return responses

chroma_db = Chroma(persist_directory="C:/Users/Devayani K/rag/db", embedding_function=OpenAIEmbeddings(),
                   collection_name='rag_db')    
similarity_threshold_retriever = chroma_db.as_retriever(search_kwargs={"k": 9})

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Initialize connection with GPT-4o
chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=0)

# Used for separating context docs with new lines
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM response generation
        web_search_needed: flag of whether to add web search - yes or no
        documents: list of context documents
    """
    question: str
    generation: str
    web_search_needed: str
    gemini_search: str
    documents: List[str]

def retrieve(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents - that contains retrieved context documents
    """
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state["question"]
    # Retrieval
    documents = similarity_threshold_retriever.invoke(question)
    #print(documents)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    by using an LLM Grader.
    If any document are not relevant to question or documents are empty - Web Search needs to be done
    If all documents are relevant to question - Web Search is not needed
    Helps filtering out irrelevant documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    # Score each doc
    filtered_docs = []
    web_search_needed = "No"
    if documents:
        for d in documents:
            score = doc_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                #web_search_needed = "Yes"
                continue
    else:
        print("---NO DOCUMENTS RETRIEVED---")
        web_search_needed = "Yes"
    if len(filtered_docs) == 0:
        web_search_needed = "Yes"
    print('web search = ', web_search_needed)
    return {"documents": filtered_docs, "question": question,
            "web_search_needed": web_search_needed}

def rewrite_query(state):
    """
    Rewrite the query to produce a better question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates question key with a re-phrased or re-written question
    """
    print("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    print(better_question)
    return {"documents": documents, "question": better_question}

def web_search(state):
    """
    Web search based on the re-written question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    # Web search
    docs = tv_search.invoke(question)
    web_results = "\n\n".join([d["content"] for d in docs])
    #print(web_results)
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}

def generate_answer(state):
    """
    Generate answer from context document using LLM
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = qa_rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question,
            "generation": generation}

def final_check(state):
    question = state["question"]
    documents = state["documents"]
    ans = state["generation"]
    score = doc_grader.invoke(
                {"question": question, "document": ans}
            )
    grade = score.binary_score
    if grade == "no":
      gemini_search = "yes"
      print('ANS DONT MATCH')
    elif grade == "yes":
      gemini_search = "no"
      print('ANS DO MATCH')
    return {"documents": documents, "question": question,
            "generation": ans, "gemini_search": gemini_search}

async def gemini(state):
    question = state["question"]
    documents = state["documents"]
    # Ask gemini api
    #genai.configure(api_key=GOOGLE_API_KEY)
    #model = genai.GenerativeModel('gemini-pro')
    #response = model.generate_content(question)

    p = question + ''' Do NOT return fictional information. Return the answer in english'''
    r = await run_all(p)
    ans = ''
    length = 0
    for i in r:
      if i != None:
        a = "".join(c for c in i if c.isascii())
        a.strip()
        if len(a) > length:
            ans = a
            length = len(ans)

    return {"documents": documents, "question": question,
          "generation": ans}

def final_print(state):
    print("---FINAL ANSWER---")
    return {"documents": state["documents"], "question": state["question"],
            "generation": state["generation"]}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    web_search_needed = state["web_search_needed"]
    if web_search_needed == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: SOME or ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REWRITE QUERY---")
        return "rewrite_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE RESPONSE---")
        return "generate_answer"
    
def need_gemini(state):
    gemini_search = state["gemini_search"]
    print('NEED GEMINI? ',gemini_search)
    if gemini_search == "yes":
        print('NEED GEMINI')
        return "gemini"
    else:
        print('NO GEMINI')
        return "final_print"
    #######################################################################

# LLM for grading
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
# Prompt template for grading
SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
                Follow these instructions for grading:
                  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
                     {document}
                     User question:
                     {question}
                  """),
    ]
)
# Build grader chain
doc_grader = (grade_prompt
                  |
              structured_llm_grader)

# Create RAG prompt for response generation
prompt = """You are an assistant for question-answering tasks.
            Use ONLY the following pieces of retrieved context to answer the question.
            Do not make up the answer unless it is there in the provided context.
            If the answer is not there in the provided context, ONLY say that you don't know the answer.
            Give a detailed answer and to the point answer with regard to the question.

            Question:
            {question}

            Context:
            {context}

            Answer:
         """
prompt_template = ChatPromptTemplate.from_template(prompt)


# create QA RAG chain
qa_rag_chain = (
    {
        "context": (operator.itemgetter('context')
                        |
                    RunnableLambda(format_docs)),
        "question": operator.itemgetter('question')
    }
      |
    prompt_template
      |
    chatgpt
      |
    StrOutputParser()
)

SYS_PROMPT1 = """Act as a question re-writer and perform the following task:
                 - Convert the following input question to a better version that is optimized for web search.
                 - When re-writing, look at the input question and try to reason about the underlying semantic intent / meaning.
             """
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT1),
        ("human", """Here is the initial question:
                     {question}
                     Formulate an improved question.
                  """,
        ),
    ]
)

# Create rephraser chain
question_rewriter = (re_write_prompt
                        |
                       llm
                        |
                     StrOutputParser())

tv_search = TavilySearchResults(max_results=5, search_depth='advanced',max_tokens=10000)

def agent():
    agentic_rag = StateGraph(GraphState)
    # Define the nodes
    agentic_rag.add_node("retrieve", retrieve)  # retrieve
    agentic_rag.add_node("grade_documents", grade_documents)  # grade documents
    agentic_rag.add_node("rewrite_query", rewrite_query)  # transform_query
    agentic_rag.add_node("web_search", web_search)  # web search
    agentic_rag.add_node("generate_answer", generate_answer)  # generate answer
    agentic_rag.add_node("final_check", final_check)  # generate answer
    agentic_rag.add_node("gemini", gemini)  # gemini
    agentic_rag.add_node("final_print", final_print)  # print
    # Build graph
    agentic_rag.set_entry_point("retrieve")
    agentic_rag.add_edge("retrieve", "grade_documents")
    agentic_rag.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
    )
    agentic_rag.add_edge("rewrite_query", "web_search")
    agentic_rag.add_edge("web_search", "generate_answer")
    agentic_rag.add_edge("generate_answer", "final_check")
    agentic_rag.add_conditional_edges(
        "final_check",
        need_gemini,
        {"gemini": "gemini", "final_print": "final_print"},
    )
    agentic_rag.add_edge("gemini", "final_print")
    agentic_rag.add_edge("final_print", END)
    # Compile
    agentic_rag = agentic_rag.compile()
    return agentic_rag 

#agentic_rag = agent()
