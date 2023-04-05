import logging
# from haystack.utils import launch_es #this line may not be needed if you have docker running
from haystack.document_stores import ElasticsearchDocumentStore # If a jupyter notebook, you can run this cell once
import os
# from haystack.utils import clean_wiki_text
from haystack.utils import convert_files_to_docs
from haystack.nodes import PreProcessor
# from pathlib import Path
# But like I said, the 'update embedding' step is taken care of if you build a proper pipeline (i.e. let's say you create an indexing pipeline with pipeline.add_node(...) for everything. If you're manually calling write_documents() then you need to do it yourself
#https://docs.haystack.deepset.ai/docs/ready_made_pipelines#searchsummarizationpipeline

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(
    host="localhost",
    username="",
    password="",
    index="nasa1",
    create_index=True,
    similarity="dot_product"
)

# document_store.delete_documents("document")

print("Document Count before writing:", document_store.get_document_count())

# ALWAYS CHECK THAT THIS PATH IS CORRECT.
path_str = "C:/Users/Max/Documents/"  #this would be the document the user uploads
# path_str = Path("C:/Users/Max/Desktop/Textbooks/AI - ML/Test-1TB")
docs = convert_files_to_docs(dir_path=path_str)
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="sentence",
    split_length=10,
    split_overlap=2,
    split_respect_sentence_boundary=False
)
all_docs = preprocessor.process(docs)

print(f"n_files_input: {len(docs)}\nn_docs_output: {len(all_docs)}")
document_store.write_documents(all_docs)
# Update the embeddings - See Tuana's comments.
# document_store.delete_index("document") # This line deletes the entire index.

print('Success. Document Count after writing:', document_store.get_document_count())
#print(document_store.get_all_documents())
