import gradio as gr

# import the .env file
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# configuration
# DATA_PATH = r"data"
# CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# upgrade model in future? use better one? use fine-tuned one?
llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

# connect to the chromadb
vector_store = PineconeVectorStore(
    index_name="project-rip",
    embedding=embeddings_model,
)

# Set up the vectorstore to be the retriever
num_results = 3
retriever = vector_store.as_retriever(search_kwargs={"k": num_results})


# formats history
def format_history(history):
    formatted_chat = ""
    for turn in history:
        # Case A: Dictionary format (Gradio 5.x+)
        # Example: {'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'}
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")
            if role == "user":
                formatted_chat += f"User: {content}\n"
            elif role == "assistant":
                formatted_chat += f"Assistant: {content}\n"

        # Case B: List/Tuple format (Gradio 4.x)
        # Example: ["User message", "Bot message", "Extra Info"]
        # We manually pick [0] and [1] to ignore the "Too many values" error
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            formatted_chat += f"User: {turn[0]}\nAssistant: {turn[1]}\n"
    return formatted_chat


# rewrites follow-up questions (based on chat history) as standalone
condense_prompt = PromptTemplate.from_template(
    "Given the following conversation and a follow up question, "
    "rephrase the follow up question to be a standalone question.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Follow Up Input: {question}\n\n"
    "Standalone Question:"
)
condense_chain = (
    condense_prompt | llm | StrOutputParser()
)  # funnels condensed prompt through llm


# call this function for every message added to the chatbot
def stream_response(message, history):
    history_str = format_history(history)
    # print(f"Input: {message}. History: {history}\n")

    if history:
        search_query = condense_chain.invoke(
            {"chat_history": history_str, "question": message}
        )
        print(f"USING HISTORY: {search_query}")
    else:
        print("NO HISTORY")
        search_query = message

    # retrieve the relevant chunks based on the formatted query
    docs = retriever.invoke(search_query)

    # add chunks to knowledge w/ sources
    knowledge = ""
    sources_list = []

    for doc in docs:
        knowledge += doc.page_content + "\n\n"

        print(f"💥METADATA: {doc.metadata}")
        source_name = doc.metadata.get("source", "Unkown Source")
        page_id = doc.metadata.get("page_id", "?")
        sources_list.append(f"{source_name} (Page ID {page_id})")

    unique_sources = list(set(sources_list))
    sources_text = "\n".join([f"- {s}" for s in unique_sources])

    # make the call to the LLM (including prompt)
    if message is not None:
        rag_prompt = f"""
        You are an assistent which answers questions based ONLY on the following context.
        While answering, you don't use your internal knowledge, 
        but solely the information in the context section.
        You don't mention anything to the user about the provided knowledge.
        If the answer is not in the context, say "I don't know."

        Context:
        {knowledge}

        Question: {search_query}
        """

        # print(rag_prompt)

        # stream response to gradio
        partial_message = ""
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

        # append sources
        if unique_sources:
            yield partial_message + f"\n\n**Sources**\n{sources_text}"


# initiate the Gradio app
chatbot = gr.ChatInterface(
    stream_response,
    textbox=gr.Textbox(
        placeholder="Ask me anything...", container=False, autoscroll=True, scale=7
    ),
    title="Project RIP Chatbot",
    description="Ask questions about [Jinx]",
    # theme="soft",
)

# launch the Gradio app
chatbot.launch()
