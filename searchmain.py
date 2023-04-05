import logging
from haystack.utils import launch_es # This line may not be needed if you have docker running
from haystack.document_stores import ElasticsearchDocumentStore
import os
launch_es()

# print(ElasticsearchDocumentStore().get_document_count())

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
    similarity="dot_product")

# docs = document_store.get_all_documents()
# print(docs)

from haystack.nodes import BM25Retriever
retriever = BM25Retriever(document_store=document_store)
# Will use this once GPT-4 Access Granted; https://docs.haystack.deepset.ai/docs/retriever#multimodal-retrieval
#from haystack.nodes import FARMReader
# Load a local model or any of the QA models on HuggingFace's model hub (https://huggingface.co/models)

#reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

from haystack.pipelines import DocumentSearchPipeline
pipeline = DocumentSearchPipeline(retriever)
query = "What is the PLD best practice in clock design?"
result = pipeline.run(query, params={"Retriever": {"top_k": 10}})
#print(result)
combined_content = "\n".join([doc.content for doc in result["documents"]])
combined_Names = ', '.join(set([doc.meta['name'] for doc in result["documents"]]))
#print(combined_content)

import openai

# Load your API key from an environment variable or secret management service
openai.api_key = "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"
prompt = "You are a friendly and verbose assistant, given this information retrieved from a database of documents:" + combined_content + "\n Respond to this question:" + query + "\nYour response:"
model = "text-davinci-003"

# Call the API
completions = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=1024,
    temperature=0.1,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

generated_text = completions.choices[0].text
print(generated_text + "\nSources: " + combined_Names)

# from haystack.pipelines import ExtractiveQAPipeline
# pipe = ExtractiveQAPipeline(reader, retriever)
#ExtractiveQA Pipe Line, not the Semantic Search Pipeline, how can we return document location?

# prediction = pipe.run(
#     query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 5}}
# )
# from haystack.utils import print_answers
#
# # Change `minimum` to `medium` or `all` to raise the level of detail
# print_answers(prediction, details="medium")

#docu_About = ElasticsearchDocumentStore.get_all_documents(document_store)

#print(docu_About)