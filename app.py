import gradio as gr

# import the .env file
from dotenv import load_dotenv

# from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from ingest import ingest_fandom_wiki

load_dotenv()

# for having nowrap input gradio textboxes
custom_css = """
/* desktop rules */

@media (min-width: 768px) {
    /* make main gradio container fill viewport height */
    .gradio-container {
        height: 100vh !important;
    }

    /* turn main column into a flex container */
    #main-col {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* other input field styling */
    .no-wrap textarea, .no-wrap input {
        white-space: nowrap !important;
        overflow-x: scroll !important;
        /* to address bug where max_lines property converts txtbx to input and causes input to be white */
        background-color: var(--input-background-fill) !important;
    }

    /* system status box styling */
    .status-box {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        min-height: 0 !important;
    }

    .status-box textarea {
        height: 100% !important;
        overflow-y: scroll !important;
    }
}

/* mobile rules (screens smaller than 768px) */
@media (max-width: 767px) {
    .status-box textarea {
        height: 400px !important; /* Give it a fixed height so it's usable */
    }
}
"""

# configuration
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# upgrade model in future? use better one? use fine-tuned one?
llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

# connect to the chromadb
vector_store = PineconeVectorStore(
    index_name="project-rip",
    embedding=embeddings_model,
)

# Set up the vectorstore to be the retriever
num_results = 8
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


# clear input fields (wrapper for ingest_fandom_wiki)
def ingest_and_clear(target_url, character_name):
    # 1. Run the actual logic
    status_msg = ingest_fandom_wiki(target_url, character_name)

    # 2. Return the status + two empty strings to clear the textboxes
    return status_msg, "", ""


# call this function for every message added to the chatbot
def stream_response(message, history):
    # handle automatic ingestion
    # if target_url:
    #     print(f"🚀 Processing URL: {target_url}")
    #     status_msg = ingest_fandom_wiki(target_url, character_name)
    #     print(f"System: {status_msg}")

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
    try:
        docs = retriever.invoke(search_query)
    except Exception:
        docs = []
    if not docs:
        yield "❌ I couldn't find any information in the database. Please upload character data first."
        return

    # add chunks to knowledge w/ sources
    knowledge = ""
    sources_map = {}

    for doc in docs:
        knowledge += doc.page_content + "\n\n"

        print(f"ℹ️ METADATA: {doc.metadata}")
        source = doc.metadata.get("source", "Unkown Source")
        chunk_char_name = doc.metadata.get("character", "Unknown Character")
        raw_sections = doc.metadata.get("section", "General Context")

        # if isinstance(raw_sections, str):
        #     raw_sections = [raw_sections]

        if source not in sources_map:
            sources_map[source] = {"character": chunk_char_name, "sections": set()}

        sources_map[source]["sections"].add(raw_sections)

        # sources_map[source].update(raw_sections)

    final_sources_lines = []
    for source, data in sources_map.items():
        char_label = data["character"]
        sections_set = data.get("sections", set())
        sorted_sections = sorted(list(sections_set))

        sections_str = (
            '", "'.join(sorted_sections) if sorted_sections else "General Context"
        )
        final_sources_lines.append(
            f'Character: {char_label}\n- Source: {source}\n- Related sections: ["{sections_str}"]\n'
        )

    sources_text = "\n".join(final_sources_lines)
    # make the call to the LLM (including prompt)
    # You don't mention anything to the user about the provided knowledge.
    if message is not None:
        rag_prompt = f"""
        You are an assistent for Project RIP.
        
        Instructions:
        1. Priority: Check the "Context" below for the answer.
        2. If the answer is in the context, answer strictly based on that, and don't mention any previously given context.
        3. Fallback: If the answer is not in the context, ignore the context completely and answer using your own general knowledge.
        4. Disclaimer: If you use your own knowledge, start the answer with: "I couldn't find exact details in the database, but..."

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
        if sources_text:
            yield partial_message + f"\n\n**Context**\n{sources_text}"


# main ui
with gr.Blocks(title="Project RIP Chatbot") as main:
    gr.Markdown("# 🪦 Project RIP: Roleplay Inference Pipeline")

    # control panel
    with gr.Row():
        with gr.Column(scale=4, elem_id="main-col"):
            chatbot_interface = gr.ChatInterface(
                fn=stream_response,
                textbox=gr.Textbox(
                    placeholder="Ask me anything...", container=False, scale=4
                ),
            )
        with gr.Column(scale=1):
            gr.Markdown("""
            **How to use:**
            1. Choose a character (or upload your own!).
            1. Paste a Wiki URL (or two!).
            2. Click **Upload** and wait for Success.
            4. Chat on the left!
            """)

            gr.Markdown("### 🚀 Upload Data")

            char_input = gr.Textbox(
                label="Character Name",
                placeholder="e.g. Jinx, Barney Stinson, etc.",
                elem_classes="no-wrap",
                scale=1,
                lines=1,
                max_lines=1,
            )
            url_input = gr.Textbox(
                label="Target Wiki URL",
                placeholder="e.g. Fandom, Wikipedia, etc.",
                elem_classes="no-wrap",
                scale=1,
                lines=1,
                max_lines=1,
            )

            ingest_btn = gr.Button("🚀 Upload Data", variant="primary", scale=1)

            ingest_status = gr.Textbox(
                label="System Status",
                value="🟢 Ready 🟢",
                interactive=False,
                scale=1,
                elem_classes="status-box",
            )

    ingest_btn.click(
        fn=ingest_and_clear,
        inputs=[url_input, char_input],
        outputs=[ingest_status, char_input, url_input],
    )

## def main
if __name__ == "__main__":
    main.launch(theme="gstaff/sketch", css=custom_css, share=True)
