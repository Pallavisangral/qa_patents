from langchain_community.document_loaders import JSONLoader
import os
from itertools import chain


class read_json_files:
    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path

    def metadata_func(self,record: dict, metadata: dict) -> dict:

        metadata["abstract"] = record.get("abstract")
        metadata["claims"] = record.get("claims")

        return metadata
    def get_data(self):
        all_data =[]
        for file in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file)
            loader = JSONLoader(file_path,
                                jq_schema='.descriptions[]',
                                content_key="paragraph_markup",
                                metadata_func=self.metadata_func)
            data = loader.load()
            all_data.append(data)
        documents = list(chain.from_iterable(all_data))
        return documents
        