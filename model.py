from langchain_community.embeddings import OllamaEmbeddings
# Using chroma vectorstore
from langchain_community.vectorstores import Chroma
# For setting up the chat template
from langchain_core.prompts import ChatPromptTemplate
# for using the groq llama3 api
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import json
from fuzzywuzzy import process

os.environ['GROQ_API_KEY'] = ''

class MyCustomError(Exception):
    def __init__(self, message):
        super().__init__(message)
        
class local_llama3_chatbot:
    def __init__(self, model, vector_dir, embedding):
        self.llm = model
        self.db = vector_dir
        self.embedding = embedding
        self.clients = []
    def load_base_assets(self):
        self.llm = ChatGroq(model_name = self.llm, temperature=0)
        self.embedding = OllamaEmbeddings(model=self.embedding)
        self.db = Chroma(collection_name="MSA_4k_chunks", 
                         embedding_function=self.embedding,
                         persist_directory=self.db)
    def parse_meta(self, info):
        text = info.strip()
        if '{' in text:
            text = text[text.find('{'):]
        print("Before conversion - "+info)
        if not text.startswith('{') and '{' not in text:
            text = '{' + text
        # Check and add '}' at the end if missing
        if not text.endswith('}') and '}' not in text:
            text = text + '}'
        dictionary = json.loads(text)
        print("after conversion - "+str(dictionary))
        return dictionary
    def fetch_chat_metadata(self, user_query):
        # function to get the required metadata from the query and also understand the nature of the
        # query( to check for comparative documents )
        system = "You are a helpful assistant who answers exactly in the format specified by the user."
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{text}")])
        chain = prompt | self.llm
        user_query = 'From the given text extract the organization name and the year as a python dictionary response with keys as client and contract_date(in the format YYYY).Do not return any other additonal messages. For unfound values return the value as "" for the ccorresponding key. Some sample client names will look like ["abc corp", "abc org", "abc inc", "abc ai"] TEXT: '+user_query
        meta_response = chain.invoke({"text":user_query})
        metadata = meta_response.content
        self.llm='llama3-8b-8192'
        self.llm = ChatGroq(model_name = self.llm)
        chain = prompt | self.llm
        user_query = 'From the given text extract the organization name and the year of the contract as a python dictionary response with keys as client and contract_date(in the format YYYY).Do not return any other additonal messages. For unfound values return the value as "" for the ccorresponding key. Some sample client names will look like ["abc corp", "abc org", "abc inc", "abc ai"] TEXT: '+user_query
        check_query = f'From the given text, check whether it is a cross-document question and return the response as "cross-document" if True and if False return response as "solo-document". Do not reurn any other additional message in your response. Cross-document question involves more than 2 organization names and asking for aggregations across organizations. TEXT: {user_query}"'
        response = chain.invoke({"text":check_query})
        q_type = response.content
        try:
            metadata = eval(metadata)
        except:
            metadata = self.parse_meta(metadata)
        return {'question_type': q_type, 'metadata': metadata}
    def model_call(self, user_query, classification):
        client = classification['metadata']['client']
        year = classification['metadata']['contract_date']
        searchtype = classification['question_type']
        metadatas = self.db.get()['metadatas']
        self.clients = [i['client'] for i in metadatas]
        correct_client = process.extractOne(client, self.clients)[0]
        print(f'client is {correct_client}')
        retriever = self.db.as_retriever(search_kwargs={"k":3})
        # You can simplify the filter dict dynamically instead of these many if statements.
        # I have done this for simplicity
        if searchtype=='solo-document' and isinstance(year, int)==False:
            print("entered - "+searchtype)
            retriever = self.db.as_retriever(search_kwargs={"k":3,
                                                        "filter":{
                                                            'client': correct_client
                                                        }})
        # elif searchtype=='cross-document' and isinstance(year,int):
        #     print("entered - "+searchtype)
        #     retriever = self.db.as_retriever(search_kwargs={"k":3,
        #                                                 "filter":{
        #                                                     'contract_date': {'$gt': f'01-01-{year}'}
        #                                                 }})
        elif searchtype=='solo-document' and isinstance(year,int):
            print("entered - "+searchtype)
            retriever = self.db.as_retriever(search_kwargs={"k":3,
                                                        "filter":{
                                                            'client': correct_client,
                                                            'contract_date': {'$gt': f'01-01-{year}'}
                                                        }})
        qa_chain_response = self.final_llm_call(user_query,retriever)
        # qa_chain_response = self.vector_searcher(user_query, correct_client)
        return qa_chain_response
    def vector_searcher(self, user_query, correct_client):
        return self.db.similarity_search(user_query, k=3,filter={'client': correct_client})
    def final_llm_call(self, user_query, retriever):
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        response = chain.invoke({"input":user_query})
        return response
if __name__ == '__main__':
    chatbot = local_llama3_chatbot('llama3-8b-8192', './db','llama3')
    query = input("query : ") # 'what is the address of shiro corp'
    chatbot.load_base_assets()
    while query!='exit':
        classification = chatbot.fetch_chat_metadata(query)
        output = chatbot.model_call(query, classification)
        # print(output)
        print("Output is : ",output['answer'])
        query = input("query : ")
    print("========================End Of Conversation=================================")