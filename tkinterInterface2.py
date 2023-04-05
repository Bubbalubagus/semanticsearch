import tkinter as tk
from tkinter import ttk
from tkhtmlview import HTMLLabel
import threading
from haystack.utils import launch_es  # This line may not be needed if you have docker running
launch_es()
from haystack.document_stores import ElasticsearchDocumentStore
import os

# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(
    host="localhost",
    username="",
    password="",
    index="nasa1",
    create_index=True,
    similarity="dot_product")

import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

file_paths = [
    "C:\\Users\\Max\\Documents\\NASA Documents for DEMO\\2022-10-24-NASA-STD-5001B-w-Change-3-Approved.pdf",
    "C:\\Users\\Max\\Documents\\NASA Documents for DEMO\\20190029153.pdf",
    "C:\\Users\\Max\\Documents\\NASA Documents for DEMO\\handbook_870922_baseline_with_change_7.pdf",
    "C:\\Users\\Max\\Documents\\NASA Documents for DEMO\\nasa-hdbk-4009a_w-chg_1.pdf",
    "C:\\Users\\Max\\Documents\\NASA Documents for DEMO\\NASA-HDBK-4008w-Chg-1.pdf",
    "C:\\Users\\Max\\Documents\\NASA Documents for DEMO\\nasa-std-4009a_w-chg_1.pdf",
    "C:\\Users\\Max\\Documents\\NASA Documents for DEMO\\NASA-STD-5012B.pdf",
    "C:\\Users\\Max\\Documents\\NASA Documents for DEMO\\SpacePacketProtocol.pdf",
]

def search_function(query):
    from haystack.nodes import BM25Retriever
    retriever = BM25Retriever(document_store=document_store)

    from haystack.pipelines import DocumentSearchPipeline
    pipeline = DocumentSearchPipeline(retriever)
    # query = query
    result = pipeline.run(query, params={"Retriever": {"top_k": 8}})
    print(result)
    print("\n______________________________________________")
    combined_content = "\n".join([doc.content for doc in result["documents"]])
    combined_Names = ', '.join(set([doc.meta['name'] for doc in result["documents"]]))
    combined_Names_array = list(set([doc.meta['name'] for doc in result["documents"]]))
    returned_document_paths = []
    for name in combined_Names_array:
        returned_document_paths.append("C:\\Users\\Max\\Documents\\NASA Documents for DEMO\\"+name)
    print(str(returned_document_paths))
    print("\n______________________________________________\n")
    print(combined_content)
    for path in returned_document_paths:
        file_name = os.path.basename(path)
        label_file = create_clickable_label(root, file_name, path)
        label_file.pack(pady=5)

    import openai
    # Load your API key from an environment variable or secret management service
    openai.api_key = "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"
    prompt = "You are a friendly and verbose assistant, given what you already know and this information retrieved from a database of documents:" + combined_content + "\n Respond to this question:" + query + "\nYour response:"
    model = "text-davinci-003"
    print(" - I'm now working on the LLM Call. -")
    # Call the API
    completions = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=500,
        temperature=0.1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    generated_text = completions.choices[0].text
    print(generated_text + "\nSources: " + combined_Names)
    print("~_____________________~\nI have finished making the LLM Call.")

    return generated_text + "\n\nSources: " + combined_Names


# Function to handle search button click
def on_search_click():
    query = entry.get()
    if query.lower() == 'quit':
        root.destroy()
    else:
        # Start the progress indicator
        start_progress_indicator()

        # Run the search function in a separate thread
        search_thread = threading.Thread(target=run_search, args=(query,))
        search_thread.start()

def open_file(path):
    os.startfile(path)

def create_clickable_label(parent, text, path):
    label = tk.Label(parent, text=text, fg="blue", cursor="hand2")
    label.bind("<Button-1>", lambda e: open_file(path))
    return label

def run_search(query):
    result = search_function(query)
    # Stop the progress indicator and update the result label
    root.after_idle(stop_progress_indicator)
    root.after_idle(result_label.config, {'text': result})

def start_progress_indicator():
    progress_indicator.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    progress_indicator.start()

def stop_progress_indicator():
    progress_indicator.stop()
    progress_indicator.place_forget()

# Create the main window
root = tk.Tk()
root.title("NASA's AI Assistant: Search Tool")
root.geometry("800x500")
root.configure(bg='#F5F5F5')
# Bind the "Enter" key to the on_search_click function
root.bind('<Return>', lambda event: on_search_click())

# Create a custom font
custom_font = ('Helvetica', 14)
custom_font_small = ('Helvetica', 10)

# HTML Code Below (not working)
# html_content = """
# <html>
# <body> </br> </br> </br> </br>
# <h3>How to use this tool:</h3>
# <p>Use to hunt for answers that are hidden across millions of documents!</p>
# <ul>
#     <li>Ask specific questions</li>
#     <li>Check the sources provided</li>
#     <li>Useful for email-search too!</li>
# </ul>
# </body>
# </html>
# """
# html_label = HTMLLabel(root, html=html_content)
# html_label.pack(fill=tk.BOTH, expand=True)
# Create input entry
entry = ttk.Entry(root, font=custom_font, width=55)
entry.pack(pady=20)

# Create search button
search_button = ttk.Button(root, text="Search", command=on_search_click)
search_button.pack(pady=5)

# Create result label
result_label = tk.Label(root, text="", wraplength=700, font=custom_font, bg='#F5F5F5', anchor='w')
result_label.pack(pady=10)

# Create a spinning progress indicator
progress_indicator = ttk.Progressbar(root, mode='indeterminate', length=100)

# Create a label to display a variable in the bottom right corner
variable_label = tk.Label(root, text="querying " + str(int(ElasticsearchDocumentStore().get_document_count())) + " docs.", bg='#F5F5F5', font=custom_font_small)
variable_label.pack(side=tk.BOTTOM, anchor=tk.SW)


# Start the main loop
root.mainloop()