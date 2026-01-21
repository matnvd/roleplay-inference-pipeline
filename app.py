import json
import os

import gradio as gr
from dotenv import load_dotenv

# from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from ingest import ingest_fandom_wiki

load_dotenv()
INDEX_NAME = "project-rip"
API_KEY = os.getenv("PINECONE_API_KEY")
CHAR_FILE = "characters.json"

# custom css for like everything
custom_css = """
/* desktop rules */

@media (min-width: 768px) {
    /* FILL VIEWPORT HIEGHT (for main gradio container) */
    .gradio-container {
        height: 100vh !important;
    }

    /* FLEX CONTAINER (for main col) */
    #main-col {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    #main-col .prose {
        font-size: 16px !important;
        line-height: 1.5 !important;
    }

    /*TAB EXPANSION (need to adjust, not working properly entirely)*/
    .expand-tabs > div:first-child {
        display: flex !important;
        width: 100% !important;
        flex-direction: row !important;
        gap: 0 !important;
    }
    .expand-tabs > div:first-child > button {
        flex-grow: 1 !important;
        flex-basis: 0 !important;
        width: 50% !important;
        justify-content: center !important;
        text-align: center !important;
    }

    /* CHARACTER LIST STYLING */
    .char-radio-group {
        max-height: 60vh !important;
        overflow-y: auto !important;
        padding: 5px;
        # background: transparent !important;
        border: none !important;
    }

    .char-radio-group .wrap {
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
    }

    .char-radio-group label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        padding: 12px 16px !important;
        background: var(--background-fill-primary) !important;
        # border: 1px solid var(--border-color-primary) !important;
        # border-radius: 12px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
    }

    .char-radio-group label:hover {
        background: var(--background-fill-secondary) !important;
        # transform: translateX(4px); /* Little nudge effect */
        border-color: var(--color-accent) !important;
    }

    .char-radio-group label.selected {
        background: var(--neutral-700) !important;
        color: white !important;
        # border-color: var(--color-accent) !important;
        font-weight: bold !important;
        # box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* 6. Hide the default radio circle/bubble */
    .char-radio-group input[type="radio"] {
        display: none !important; /* hides bubble */
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
vectorStore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings_model,
)
pc = Pinecone(api_key=API_KEY)

# Set up the vectorstore to be the retriever
NUM_RESULTS = 8

########################################
# MANAGING MULTIPLE CHARACTERS #
########################################


# load character set from local json
def load_registry():
    if os.path.exists(CHAR_FILE):
        with open(CHAR_FILE, "r") as f:
            return list(json.load(f))
    else:
        print("🔴 Err: character registry path file does not exist")
    return list()


# add character to json
def save_to_registry(new_char):
    chars = load_registry()
    chars.append(new_char)
    with open(CHAR_FILE, "w") as f:
        json.dump(list(chars), f)
    return chars


# format characters into md and return char_display
def format_char_list(char_list):
    if not char_list:
        return "No characters found locally."
    chars = list(char_list)
    return "### 🎭 Available Characters:\n" + ", ".join([f"`{char}`" for char in chars])


# return unique characters in db to radio component
def get_character_choices():
    chars = load_registry()
    return list(chars)


# default to first choice
def get_first_choice():
    choices = get_character_choices()
    return choices[0] if choices else None


def refresh_list(cur_selection=None):
    choices = get_character_choices()

    if cur_selection and cur_selection in choices:
        new_val = cur_selection
    elif choices:
        new_val = choices[0]
    else:
        new_val = None

    if choices:
        default_val = choices[0]
        return gr.Radio(choices=choices, value=new_val, interactive=True)
    else:
        return gr.Radio(
            choices=["Please upload character data"],
            value=None,
            interactive=False,
        )


# bc pinecone unique metadata field search dne, use dummy vector to scan entire db for unique values, sorts list too
def resync_characters_scan():
    try:
        index = pc.Index(INDEX_NAME)

        dummy_vector = [0.0] * 1536
        query_response = index.query(
            vector=dummy_vector,
            top_k=10000,  # max limit should be 10k (should grab all characters)
            include_metadata=True,
            include_values=False,
        )

        unique_chars = set()
        for match in query_response["matches"]:
            if "metadata" in match and "character" in match["metadata"]:
                unique_chars.add(match["metadata"]["character"])

        # Save this fresh scan to the file
        with open(CHAR_FILE, "w") as f:
            json.dump(list(unique_chars), f)

        new_choices = sorted(list(unique_chars))
        default_val = new_choices[0] if new_choices else None

        # return new list + status update
        return format_char_list(
            unique_chars
        ), f"✅ Resync Complete. Found {len(unique_chars)} characters."

    except Exception as e:
        return f"⚠️ Error during resync: {str(e)}"


########################################
# HISTORY/SOURCING #
########################################


# formats history
def format_history(history):
    formatted_chat = ""
    for turn in history:
        # in the case its dictionary format
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")
            if role == "user":
                formatted_chat += f"User: {content}\n"
            elif role == "assistant":
                formatted_chat += f"Assistant: {content}\n"

        # in the case its a list/tuple format
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
    # run logic for ingesting
    status_msg = ingest_fandom_wiki(target_url, character_name)

    # save character to registry and update list
    save_to_registry(character_name)

    # return the status + two empty strings to clear the textboxes + updated character list
    return status_msg, "", "", refresh_list()


# current char selection
def update_char_ui(selected_char):
    return selected_char, selected_char


# call this function for every message added to the chatbot
def stream_response(message, history, selected_char):
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

    # filtering per selected character
    filter_dict = None
    if selected_char and selected_char != "All Characters":
        print(f"🎯 Filtering for character: {selected_char}")
        filter_dict = {"character": selected_char}

    # retrieve the relevant chunks based on the formatted query
    try:
        docs = vectorStore.similarity_search(
            search_query, k=NUM_RESULTS, filter=filter_dict
        )
    except Exception:
        docs = []
    if not docs:
        if selected_char:
            yield f"❌ I couldn't find any information for '{selected_char}'. Please upload character data first."
        else:
            yield "Please upload character data first."
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


########################################
# MAIN FRONTENT/UI #
########################################

with gr.Blocks(title="Project RIP Chatbot") as main:
    gr.Markdown("# 🪦 Project RIP: Roleplay Inference Pipeline")

    # cur char
    char_state = gr.State("All Characters")

    # control panel
    with gr.Row():
        with gr.Column(scale=4, elem_id="main-col"):
            chatbot_interface = gr.ChatInterface(
                fn=stream_response,
                textbox=gr.Textbox(
                    placeholder="Ask me anything...", container=False, scale=4
                ),
                additional_inputs=[char_state],
            )

            current_char_display = gr.Textbox(
                label="Active Character",
                value="No Character Available",
                interactive=False,
                lines=1,
                # max_lines=1,
                elem_id="active-char-box",
            )
        with gr.Column(scale=1):
            with gr.Tabs(elem_classes=["expand-tabs"]):
                # tab 1: initial uploading character section
                with gr.Tab("🚀 Upload Context", scale=1) as upload_tab:
                    gr.Markdown("""
                    **How to use:**
                    1. Upload your **own** character (or **choose** an existing one!).
                    1. Paste a **Wikipedia** or **Fandom Wiki** URL (or two!).
                    2. Click **Upload** and wait for Success.
                    4. Chat on the left!
                    """)

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

                # tab 2
                with gr.Tab("🎭 Characters", scale=2) as char_tab:
                    char_selector = gr.Radio(
                        label="Select Character to Chat With",
                        choices=[],
                        value=None,
                        interactive=True,
                        elem_classes=["char-radio-group"],
                        container=False,
                    )
                    refresh_btn = gr.Button("🔄 Resync Database", size="sm")

            # ingestion status + general stati updates
            system_status = gr.Textbox(
                label="System Status",
                value="🟢 Ready 🟢",
                interactive=False,
                scale=1,
                elem_classes="status-box",
            )

    # events
    char_selector.change(
        fn=update_char_ui,
        inputs=char_selector,
        outputs=[char_state, current_char_display],
    )

    ingest_btn.click(
        fn=ingest_and_clear,
        inputs=[url_input, char_input],
        outputs=[system_status, char_input, url_input, char_selector],
    )

    refresh_btn.click(
        fn=resync_characters_scan, inputs=None, outputs=[char_selector, system_status]
    )

    # makes sure to get character list every time its clicked
    char_tab.select(fn=refresh_list, inputs=[char_state], outputs=char_selector)

    main.load(fn=refresh_list, inputs=None, outputs=[char_selector])

## def main
if __name__ == "__main__":
    main.launch(theme="gstaff/sketch", css=custom_css, share=True)
