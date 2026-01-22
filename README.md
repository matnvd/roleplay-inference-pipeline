# Project R.I.P: A Roleplay Inference Pipeline
A retrieval-augmented generation engine for high-fidelity persona simulation.

Project R.I.P. is a RAG pipeline architecture that rips (haha get it) data from various content forms to bring to life (haha get it get it) coherent, accurate, role-playing persona agents through a Python Gradio app hosted on Hugging Face Spaces (If you have an access key, you can try it; otherwise watch the demo... I don't have unlimited API credits).

It is an ingestion system that scrapes Wiki data (currently supporting Fandom Wiki and Wikipedia links), chunks HTML while appending relevant source metadata, and stores the vector embeddings in a vector database in Pinecone.

There is a system prompt framework that works to enforce strict character and source-retrieval constraints, persistent memory for context retention, and the ability to generate and demarcate between multiple characters and their sources. 

Live Demo: 
https://huggingface.co/spaces/mathiasnvd/project-rip

Video Demo:

<iframe width="644" height="400" 
    src="https://www.youtube.com/embed/2Y4AmMF2oZM?si=hF2Go23ietXc0V5t" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen 
    style="border-radius: 12px;">
</iframe>
