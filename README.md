# Project R.I.P: A Roleplay Inference Pipeline
A retrieval-augmented generation engine for high-fidelity persona simulation.

Project R.I.P. is a \RAG pipeline architecture that rips (haha get it) data from various content forms to bring to life (haha get it...) coherent, accurate, role-playing persona agents through a Python Gradio app (need access key).

It is an ingestion system that scrapes Wiki data (currently tested on fandom wikis and [soon] wikipedia links), chunks HTML while appending relevant source metadata, and stores the vector embeddings in a vector database in Pinecone.

There is a system prompt framework that works to enforce strict character and source-retrieval constraints, persistent memory for context retention, and the ability to generate and demarcate between multiple characters and their sources.