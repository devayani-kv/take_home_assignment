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
from critical_role import *

import re
import g4f
import asyncio
import queries

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
#chroma_db = Chroma.from_documents(documents=docs,collection_name='rag_db',
            #embedding=embedding, collection_metadata={"hnsw:space": "cosine"}, persist_directory="db")
#chroma_db.persist()
chroma_db = Chroma(persist_directory="C:/Users/Devayani K/rag/db", embedding_function=OpenAIEmbeddings(),
                   collection_name='rag_db')

similarity_threshold_retriever = chroma_db.as_retriever(search_kwargs={"k": 8})
#print(docs)

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
    

query = '''Has he Performed in a leading or critical role for organizations or establishments that have a distinguished reputation?
            Emphasis on the word "leading" and "critical", and pay attention to whether the organizations have a distinguished reputation
            If yes, Return the his EXACT title at the organization, not just engineering leader unless nothing else is mentioned.
            Return the answer in this exact format:

            [Name of person], [His exact title at the organization] and the [name of the organization]
            for each of the organizations and the critical roles he has held at these organizations'''
ans = qa_rag_chain(query)
print(ans)

question = '''From the above context, extract the company name and the role of Jiten Oswal at the company.
            Return the answer as a dictionary with the company name as the key and the role in that company as the value.'''
ans = simple_chain(question,ans)
json_obj = json.loads('{'+re.findall(r"\{([^}]*)\}", ans)[0]+'}')
with open('orgs_designation.json','w+') as f:
    json.dump(json_obj,f)

dt = []
with open('orgs_designation.json','r') as f:
    json_obj = json.load(f)
for k,v in json_obj.items():
    query = '''What are the products, projects, funds or teams lead by Jiten Oswal when he worked at {} as a {}?
            #Return the answer as a dictionary with the key as {} and the value as the list of products, projects, funds or teams'''
    qq = query.format(k,v,k)
    print(qq)
    ans = qa_rag_chain(query)
    obj = json.loads('{'+re.findall(r"\{([^}]*)\}", ans)[0]+'}')
    dt.append(obj)
with open('orgs_projects.json','w+') as f:
    json.dump(obj,f)

with open('orgs_designation.json','r') as f:
    json_obj = json.load(f)

agentic_rag = agent()
with open('orgs_projects.json','r') as f:
    proj = json.load(f)
    
async def final():
    for k,v in json_obj.items():
        #if k != 'SoftBank' :
            #continue
        ans = ''
        b = False
        for i in range(0,8):
            if i == 0:
                ans += '\n'
                ans += '*************************** Affidavits from key personnel ***************************'
                ans += '\n'
            if i == 3:
                ans += '\n'
                ans += '*************************** Organization Profiles and Achievements *****************************'
                ans += '\n'
            if i == 6:
                ans += '\n'
                ans += '*************************** Documentation Of Roles and Contributions ******************************'
                ans += '\n'
            if i == 8:
                ans += '\n'
                ans += '*************************** Project Descriptions and Outcomes ********************************'
                ans += '\n'
            q = queries.query_list[i]
            if i<=5:
                q = q.format(k)
            elif i>5 and i<=7:
                q = q.format(v,k)
             
            print(q)
            response = await agentic_rag.ainvoke({"question": q})
            if response["generation"].lower() != 'i dont know the answer.':
                ans += '\n'
                ans += response["generation"]
                ans += '\n'

        ans += '*************************** Project Descriptions and Outcomes ********************************'
        l = proj[k]
        #print('L = ',l)
        for i in range(8,10):
            q = queries.query_list[i]
            for j in l:
                print('j ####################################### = ',j)
                #query = q.format(j,j)
                #print(query)
                if i == 8:
                    query = q.format(j,k)
                    response = await agentic_rag.ainvoke({"question": query})
                    ans += '\n'
                    ans += response["generation"]
                    ans += '\n'
                elif i == 9:
                    query = q.format(j,j,k)
                    resp = qa_rag_chain(query)
                    ans += '\n'
                    s = 'Specific contrubutions to '+j
                    ans += s
                    ans += '\n'
                    ans += resp
                    ans += '\n'
        print(ans)
        with open(k+'_info.txt','w') as f:
            f.write(ans)
        print('###################################################################################################################')
        #break

asyncio.run(final())

    

