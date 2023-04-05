import logging
# from haystack.utils import launch_es #this line may not be needed if you have docker running
from haystack.document_stores import ElasticsearchDocumentStore
import os
from haystack.utils import convert_files_to_docs

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

host = os.environ.get("ELASTICSEARCH_HOST", "localhost")
clientIndex = "papercup"  # You CAN add numbers to a clientIndex for document_store indexes
clientIndex = clientIndex.lower()
document_store = ElasticsearchDocumentStore(
    host="localhost",
    username="",
    password="",
    index= clientIndex,
    create_index=True,
    similarity="dot_product"
)

print("Document Count before writing:", document_store.get_document_count())

# ALWAYS CHECK THAT THIS PATH IS CORRECT.
path_str = "C:/Users/Max/pythonProject/pythonProject"

docs = convert_files_to_docs(dir_path=path_str, split_paragraphs=True)
document_store.write_documents(docs)
print('Success. Document Count after writing:', document_store.get_document_count())
# document_store.delete_documents(clientIndex)
# print('Success. Document Count after deleting:', document_store.get_document_count())
# print(document_store.get_all_documents())
