# Project R.I.P: A Roleplay Inference Pipeline
A retrieval-augmented generation engine for high-fidelity persona simulation.

Project R.I.P. is a RAG pipeline architecture that rips (haha get it) data from various content forms to bring to life (haha get it get it) coherent, accurate, role-playing persona agents through a Python Gradio app hosted on Hugging Face Spaces.

The project addresses the current issues of inaccurate character portrayal, especially of niche characters, by replacing standard training data with a dynamic ingestion system. It scrapes Wiki data (currently supporting Fandom Wiki and Wikipedia links), chunks HTML while appending relevant source metadata, and stores the vector embeddings in a vector database in Pinecone.

On the inference side, I built a system prompt framework that works to enforce strict character and source-retrieval constraints, persistent memory for context retention, and the ability to generate and demarcate between multiple characters and their sources. 

Live Demo (Watch the demo or try it if you have an access key! ...I don't have unlimited API credits): 
https://huggingface.co/spaces/mathiasnvd/project-rip

Video Demo:

[![Youtube Video](https://github.com/user-attachments/assets/507112c6-57a6-4ccc-b914-947484073c7c)](https://youtu.be/2Y4AmMF2oZM)
