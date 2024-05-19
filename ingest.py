# Used to load the PDF files from the source_files directory into langchain documents
from langchain_community.document_loaders import PyPDFLoader
# Used to recursively chunk the langchain document's page_content into appropriate size based on context-performance and LLM based on a
# list of characters
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Using ollama for serving llama3 embeddings
from langchain_community.embeddings import OllamaEmbeddings
# Using chroma vectorstore
from langchain_community.vectorstores import Chroma
# For setting up the chat template
from langchain_core.prompts import ChatPromptTemplate
# for using the groq llama3 api
from langchain_groq import ChatGroq
import os
import subprocess
import json

os.environ['GROQ_API_KEY'] = ''

class MyCustomError(Exception):
    def __init__(self, message):
        super().__init__(message)
        
class local_pdf_gpt_ingester:
    def __init__(self, path, embedding_model, vectorstore, images=True):
        self.path = path
        # since we are using ollama, the model should have already been served, if not then we can throw an error or pull the model oursellves
        self.embeddings = embedding_model
        # here we are only using the conditions for Chroma, but can be extended
        self.vectorstore = vectorstore
        self.images = images
    def extract_file_metadata(self, context):
        try:
            llm = ChatGroq(model_name = 'llama3-8b-8192')
            system = "You are a helpful assistant.From the given text extract the client name, service provider name and the date of the contract as a JSON response with keys as client, service_provider and contract_date(in the format dd-mm-YYYY). Do not return any other additonal messages."
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{text}")])
            chain = prompt | llm
            response = chain.invoke({"text":context})
            return response.content
        except Exception as e:
            print(e)
            return False
    def parse_meta(self, info):
        text = info.strip()
        if not text.startswith('{') and '{' not in text:
            text = '{' + text
        # Check and add '}' at the end if missing
        if not text.endswith('}') and '}' not in text:
            text = text + '}'
        dictionary = json.loads(text)
        return dictionary
    def pdf_to_documents(self, **kwargs):
        # using the langchain pdf loader for loading the documents and converting images to flat text( only OCR happens, not an image
        # embeddding) - We cannot use this loader for multimoddal content (we can use the pypdf library to extract images and text separately
        # or use the unstructured.io library)
        files = os.listdir(self.path)
        files = [f"{self.path}/{f}" for f in files if f.endswith(".pdf")]
        return_chunks = []
        for f in files:
            docs = PyPDFLoader(f,extract_images = self.images)
            # setting up the splitter (here the splitter is configured to work with llama3, need to change appropriately)
            splitter = RecursiveCharacterTextSplitter(chunk_size = kwargs.get('chunk_size', 3800),
                                                                    chunk_overlap = kwargs.get('chunk_overlap',50),
                                                                    separators = kwargs.get('separators',["\n\n","\n"," ","."]),
                                                                    is_separator_regex=kwargs.get('is_separator_regex', False)
                                                                    )
            # splitting the documents into appropriate chunks and returning the chunked content
            doc_chunks = docs.load_and_split(text_splitter = splitter)
            my_chunk = doc_chunks[0].page_content
            meta_info = self.extract_file_metadata(my_chunk)
            meta_info = self.parse_meta(meta_info)
            return_chunks.append({'doc':[i.page_content for i in doc_chunks], 'metadata': meta_info})
        return return_chunks
    def pull_model_to_ollama(model_name):
         # Construct the command
        command = ["ollama", "pull", model_name]
        try:
            # Execute the command
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Print the standard output and error
            print("Output:\n", result.stdout)
            print("Error (if any):\n", result.stderr)
            if result.stderr:
                return False
            else:
                return True
            
        except subprocess.CalledProcessError as e:
            # Handle errors in the command execution
            print(f"An error occurred while pulling the model '{model_name}':")
            print("Return code:", e.returncode)
            print("Output:\n", e.output)
            print("Error:\n", e.stderr)
    def load_embeddings(self):
        # need to check whether the embeddings are already present, if not we have to validate and pull the model into our ollama
        try:
            # find all the llama3 models here - https://ollama.com/library/llama3:8b
            # we  are using the default connection and parameters for ollama emeddings
            self.embeddings = OllamaEmbeddings(model=self.embeddings)
        except:
            model_check = self.pull_model_to_ollama(self.embeddings)
            if model_check:
                self.embeddings = OllamaEmbeddings(model=self.embeddings)
            else:
                raise MyCustomError(f"The provided model name - {self.embeddings} is invalid. Find the list of supported llama3 models here - https://ollama.com/library/llama3:8b")
    def embed_and_store(self, documents):
        # we are going to use the ollama embeddings with Chroma store here
        if self.vectorstore=='chroma':
            try:
                # default methodis get_or_create collection, so if the name already exists this will append
                # and not overwrite that content
                self.vectorstore = Chroma(collection_name="MSA_4k_chunks", 
                                        embedding_function=self.embeddings,
                                        persist_directory="./db")
            except Exception as e:
                raise MyCustomError(f"Chroma DB error - {e}")
        
        for doc in documents:
            metas = [doc['metadata'] for i in doc['doc']]
            # optionally you can also IDs. passing metadata can be donw like this, or you can directly
            # add the metadata to the document object itself. No specific reason for me doing this.
            # adding metadata also allows easy updation and deletion, we can always check if a document
            # is present or absent based on the metdata, right.
            self.vectorstore.add_texts(texts = doc['doc'],
                                       metadatas = metas)
         
        print("The documents have been embedded and stored in the vector database")
if __name__=='__main__':
    msa_bot = local_pdf_gpt_ingester("source_files/", "llama3", "chroma")
    msa_docs = msa_bot.pdf_to_documents()
    msa_bot.load_embeddings()
    msa_bot.embed_and_store(msa_docs)
        
                