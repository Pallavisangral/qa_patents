# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from vector_model import vector_model
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ["api_key"],temperature =0.3)


class patent_retrieval:
    #load vector store
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en",encode_kwargs={'normalize_embeddings': True})
    def load_local_vector_store(self):
        vector_store = FAISS.load_local("vector_db", self.embeddings)
        
        return vector_store
    
    def get_retriever(self):
        db = self.load_local_vector_store()
        # create a chain to answer questions
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})
        # qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever)
        system_prompt = (
    "you are given description and you have to give summarized answer."
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
        prompt = ChatPromptTemplate.from_messages( [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        return chain

        # chain = RetrievalQA.from_chain_type(
        # llm=llm, chain_type="stuff", retriever=retriever,input_key = "query", return_source_documents=True)
        # return chain

# obj = patent_retrieval()
# chain = obj.get_retriever()
# query = "data processing system"
# output = chain.invoke({"input": query})
# print(output['answer'])





# def get_qa_chain():
#     # vector_store =vector_model.create_vector_db()
#     obj = vector_model()
#     embeddings = obj.load_tokenizer()
#     vector_store = FAISS.load_local("vector_db", embeddings)
#     retriever =vector_store.as_retriever(score_threshold =0.7)
    
#     prompt_template =""" Given the following context and question, generate an answer based on this context.In the answer try to provide as much as text
#     possible from "response" section in the source document. If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
#     CONTEXT : {context}

#     QUESTION : {question}

#     """

#     PROMPT = PromptTemplate(
#         template = prompt_template,
#         input_variables = ["context","question"]
#     )

#     chain = RetrievalQA.from_chain_type(llm=llm,
#                 chain_type = "stuff",
#                 retriever = retriever,
#                 input_key = "query",
#                 return_source_documents = True,
#                                         chain_type_kwargs = {'prompt':PROMPT}

#                                         )
#     return chain

# chain = get_qa_chain()
# print(chain('what is data processing and processing system in a patent document?'))