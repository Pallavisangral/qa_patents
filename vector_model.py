from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from read_json import read_json_files
import os


cur_path = os.path.dirname(__file__)
folder_path = os.path.join(cur_path, "patent_jsons")

vectordb_file_path='vector_db'

class vector_model:

    def __init__(self) -> None:
        self.huggingface_embeddings=HuggingFaceEmbeddings
        self.store= FAISS 
    

    def load_tokenizer(self):
        print('Loading Tokenizer')
        huggingface_embeddings = self.huggingface_embeddings(
            model_name="BAAI/bge-large-en",
            encode_kwargs={'normalize_embeddings': True}
        )
        print('Tokenizer loaded')
        return huggingface_embeddings
        
    def create_vector_db(self):
        obj = read_json_files(folder_path)
        docs = obj.get_data()
        embeddings =self.load_tokenizer()
        vector_db =self.store.from_documents(documents=docs, embedding=embeddings)
        vector_db.save_local(vectordb_file_path)
        # vector_store = FAISS.load_local("vector_db", embeddings)
        return vector_db

# obj = vector_model()
# vector_db=obj.create_vector_db()

