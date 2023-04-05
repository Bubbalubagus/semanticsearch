import logging
from haystack.utils import convert_files_to_docs
from haystack.nodes import PreProcessor, EmbeddingRetriever
from haystack.document_stores import WeaviateDocumentStore

import json

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

document_store = WeaviateDocumentStore(recreate_index=True)
document_store.delete_documents()

model_format='sentence_transformers'
embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base"
use_gpu=True

def pre_embedder(docs):
    print('Running the pre-embedding')
    retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model=embedding_model,
    model_format=model_format,
    use_gpu=True,
    batch_size=100
    )
    embeds = retriever.embed_documents(docs)
    for doc, emb in zip(docs,embeds):
        doc.meta['name'] = ((doc.meta['name']).rsplit(".", 1)[0]).replace("_", " ")
        doc.embedding = emb
    return docs


doc_dir = "reports"
all_docs = convert_files_to_docs(dir_path=doc_dir)

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="sentence",
    split_length=10,
    split_overlap=2
)

docs = preprocessor.process(all_docs)

docs = pre_embedder(docs)

reports = []

for doc in docs:
    doc_content = json.loads(doc.to_json())
    reports.append(doc_content)

document_store.write_documents(documents=reports, batch_size=100)