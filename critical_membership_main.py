import os
os.environ["OPENAI_API_KEY"] = "sk-proj-0GT7K5HQiFRZ42GQHo8rT3BlbkFJotR609bAI4DKyLatRsGw"
os.environ["TAVILY_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBENeAhSmK2pdyJds1YGj0XyLZcl2k5ATc"

import numpy as np
import pandas as pd
import json
import re
import operator 
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import google.generativeai as genai
#from langchain_chroma import Chroma
from langchain_community.vectorstores.chroma import Chroma
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
from critical_membership import *

import re
import g4f
import asyncio
import queries1

embedding = OpenAIEmbeddings()

loader = DirectoryLoader('', glob="./*.pdf", loader_cls=PyPDFLoader)
pages = loader.load()

for i in range(len(pages)):
    pages[i].page_content = re.sub(r'\n+', ' ',pages[i].page_content.strip())

new_pages = []
new_pages.append(pages[0])
new_pages.append(pages[1])
pgs = ''
for i in range(2, len(pages)):
    pgs += pages[i].page_content
pgs = Document(page_content=pgs)
new_pages.append(pgs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500,
)
docs = text_splitter.split_documents(new_pages)
chroma_db1 = Chroma.from_documents(documents=docs,collection_name='rag_db1',
            embedding=embedding, collection_metadata={"hnsw:space": "cosine"}, persist_directory="C:/Users/Devayani K/rag/db1")
#chroma_db1.persist()

#chroma_db1 = Chroma(persist_directory="C:/Users/Devayani K/rag/db1", embedding_function=OpenAIEmbeddings(),
#                   collection_name='rag_db1')

similarity_threshold_retriever = chroma_db1.as_retriever(search_kwargs={"k": 8})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=0)

def qa_rag_chain(query):
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

    top3_docs = similarity_threshold_retriever.get_relevant_documents(query)
    #print(top3_docs)
    result = qa_rag_chain.invoke(
        {"context": top3_docs, "question": query}
    )
    return result

def simple_chain(question, ans):
    prompt1 = """You are an assistant for question-answering tasks.
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
    prompt_template = ChatPromptTemplate.from_template(prompt1)

    # create QA RAG chain
    simple_chain = (
        {
            "context": operator.itemgetter('context'),
            "question": operator.itemgetter('question')
        }
        |
        prompt_template
        |
        chatgpt
        |
        StrOutputParser()
    )

    result = simple_chain.invoke(
        {"context": ans, "question": question}
    )
    #print(result)
    return result
    
query = '''Has he held any memberships/fellowships in associations that require oustanding achievements
            If yes, return the asnwer in this format

            [Name of person], [name of the organization where he holds a membership]
            for each of the organizations'''
#ans = qa_rag_chain(query)
#print(ans)

question = '''From the above context, extract the name of the organization and the name of the membership/fellowship.
            Return the answer as a dictionary with the name of the organization as the key and the name of the membership/fellowship as the value.'''
#ans = simple_chain(question,ans)
#json_obj = json.loads('{'+re.findall(r"\{([^}]*)\}", ans)[0]+'}')
#print(json_obj)


with open('orgs_membership.json','r') as f:
    json_obj = json.load(f)

agentic_rag = agent()

#Stanford Web3 & AI Research Group
async def trial():
    query = '''What is the specific criteria that one needs to satidy for membership in Stanford Web3 & AI Research Group?'''
    response = await agentic_rag.ainvoke({"question": query})
    print(response["generation"])



async def final():
    for k,v in json_obj.items():
        ans = ''
        #if k != "Delta Analytics":
            #continue
        for i in range(len(queries1.query)):
            if i == 0:
                ans += '########################### Association Profiles ##################################'
                ans += '\n'
            if i == 3:
                ans += '\n'
                ans += '########################### Membership Requirements ##################################'
                ans += '\n'
            q = queries1.query[i]
            if 'Stanford' in k:
                q = q.format(k)
            else:
                q = q.format(k + ' ' + v)
            print(q)
            response = await agentic_rag.ainvoke({"question": q})
            ans += '\n'
            if response["generation"].lower() != "i don't know the answer":
                ans += response["generation"]
                ans += '\n'

        ans += '#################### Affidavits from Association Officials ##########################'
        query7 = '''what is the rigorous membership criteria to be a {} member?'''
        if 'Stanford' in k:
            query7 = query7.format(k)
        else:
            query7 = query7.format(k + ' ' + v)
        response1 = await agentic_rag.ainvoke({"question": query7})
        ans += '\n'
        ans += response1["generation"]
        ans += '\n'

        query8 = '''Which of these criterias does Jiten Oswal meet 
        
                    {}?'''
        query8 = query8.format(response1["generation"])
        response = qa_rag_chain(query8)
        #response = await agentic_rag.ainvoke({"question": query8})
        if response.lower() != "i don't know the answer.":
            ans += '\n'
            #ans += response["generation"]
            ans += response
            ans += '\n'
        else:
            query = queries1.query_extra
            query = query.format(v)
            response = qa_rag_chain(query)
            ans += '\n'
            #ans += response["generation"]
            ans += response
            ans += '\n'

        query9 = '''What is the role of recognized experts in the selection process for {}? If you cant find the answer say I don't know'''
        if 'Stanford' in k:
            query9 = query9.format(k)
        else:
            query9 = query9.format(k + ' ' + v)
        response = await agentic_rag.ainvoke({"question": query9})
        ans += '\n'
        ans += response["generation"]
        ans += '\n'

        print(ans)
        with open(k+'_info.txt') as f:
            f.write(ans)
        
        #break
    return ans

aa = asyncio.run(final())
with open('stanford_info.txt') as f:
    f.write(aa)
#asyncio.run(trial())
