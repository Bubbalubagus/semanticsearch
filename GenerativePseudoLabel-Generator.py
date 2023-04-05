from haystack.nodes import PseudoLabelGenerator
from haystack.nodes.retriever import EmbeddingRetriever
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes.question_generator.question_generator import QuestionGenerator
from haystack.nodes.label_generator.pseudo_label_generator import PseudoLabelGenerator

document_store = ElasticsearchDocumentStore(
    host="localhost", username="", password="",
    index="document", create_index=True, similarity="dot_product")

retriever = EmbeddingRetriever(document_store=document_store,
                               embedding_model="sentence-transformers/msmarco-distilbert-base-tas-b",
                               model_format="sentence_transformers",
                               max_seq_len=200)
document_store.update_embeddings(retriever)

qg = QuestionGenerator(model_name_or_path="doc2query/msmarco-t5-base-v1", max_length=64, split_length=200, batch_size=12, use_gpu=True)
plg = PseudoLabelGenerator(qg, retriever)
output, _ = plg.run(documents=document_store.get_all_documents())
retriever.train(output["gpl_labels"]) # <-- Now I have an adopted retriever.
# ^ Now I have an adopted retriever. This line may be unnecessary (L20)

from haystack.nodes import BM25Retriever
retrieverQA = BM25Retriever(document_store=document_store)

from haystack.nodes import FARMReader
# Load a local model or any of the QA models (https://huggingface.co/models)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

from haystack.pipelines import ExtractiveQAPipeline
pipe = ExtractiveQAPipeline(reader, retriever)
from haystack.utils import print_answers

#print(output["gpl_labels"])
for x in output['gpl_labels']:
    queryFromGPL = x['question']
    prediction = pipe.run(
        query=queryFromGPL, params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}}
    )
    print_answers(prediction, details="minimum")

# document_store = ElasticsearchDocumentStore(
#     host="localhost", username="", password="",
#     index="document", create_index=True, similarity="dot_product")
#
# retriever = EmbeddingRetriever(document_store=document_store,
#                                embedding_model="sentence-transformers/msmarco-distilbert-base-tas-b",
#                                model_format="sentence_transformers",
#                                max_seq_len=2000)
#
#
# qg = QuestionGenerator(model_name_or_path="doc2query/msmarco-t5-base-v1")
# plg = PseudoLabelGenerator(qg, retriever)
# train_examples = []
#
# for idx, doc in enumerate(document_store):
#     output_samples = plg.run(documents=[doc])
#     for item in output_samples:
#         train_examples.append(item)
#
# print(train_examples)
# for x in train_examples['gpl_labels']:
#     print(x['question'])