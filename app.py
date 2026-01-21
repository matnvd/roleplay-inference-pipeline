import os
import re
import uuid

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
# CHAR_FILE = "characters.json"
GLOBAL_SESSION_ID = "demo_roster"

# custom css for like everything
custom_css = """
/* desktop rules */

@media (min-width: 768px) {
    /* CHAT WINDOW STYLING */
    .gradio-container {
        height: 100vh !important;
    }

    /* FLEX CONTAINER (for main col) */
    #main-col {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    #chat-window {
        background: transparent !important;
        background-color: transparent !important;
        # border: none !important;
        height: 65vh !important; /* Adjust height as needed */
    }

    #chat-window .block, #chat-window .wrap, #chat-window .bubble-wrap {
        background: transparent !important;
        border: none !important;
    }

    /* the bot text area */
    #chat-window .message-wrap {
        gap: 15px; /* Spacing between bubbles */
    }

    /* message bubbles */
    #chat-window .message {
        background-color: rgba(30, 30, 30, 0.4) !important;
        backdrop-filter: blur(8px) !important;
        # border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
        border-radius: 15px !important;
    }

    /* user bubbles */
    #chat-window .message.user {
        # border-bottom-right-radius: 2px !important;
        background-color: rgba(60, 60, 80, 0.5) !important;
    }

    /* bot bubbles */
    #chat-window .message.bot {
        border-bottom-left-radius: 2px !important;
        background-color: rgba(40, 40, 40, 0.5) !important;
    }

    #main-col .prose {
        font-size: 16px !important;
        line-height: 1.5 !important;
    }

    /* INPUT ROW STYLING */
    #input-row {
        align-items: stretch !important;
        gap: 8px !important;
    }
    
    #send-btn {
        height: 100% !important;
        margin: 0 !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
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
        /*background: transparent !important;*/
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
        /*border: 1px solid var(--border-color-primary) !important;*/
        /*border-radius: 12px !important;*/
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
    }

    .char-radio-group label:hover {
        background: var(--background-fill-secondary) !important;
        /*transform: translateX(4px); */
        border-color: var(--color-accent) !important;
    }

    .char-radio-group label.selected {
        background: var(--neutral-700) !important;
        color: white !important;
        /*border-color: var(--color-accent) !important;*/
        font-weight: bold !important;
        /*box-shadow: 0 4px 6px rgba(0,0,0,0.1);*/
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
        height: 85px !important;
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
# MANAGING MULTIPLE SESSIONS #
########################################


def get_session_id():
    new_id = str(uuid.uuid4())
    print(f"\n\n🆕 New Session Started: {new_id}")
    return new_id


def update_radio_list(global_list, session_list, selected=None):
    # start w/ sorted global list
    combined_choices = sorted(list(global_list))

    # append session characters by order of upload
    for char in session_list:
        if char not in combined_choices:
            combined_choices.append(char)

    if not combined_choices:
        return gr.Radio(
            choices=["Please upload character data"], value=None, interactive=False
        )

    # Keep selection if valid, else pick first
    new_val = (
        selected if (selected and selected in combined_choices) else combined_choices[0]
    )

    return gr.Radio(choices=combined_choices, value=new_val, interactive=True)


# scan pinecone for global characters only
def fetch_global_characters():
    try:
        index = pc.Index(INDEX_NAME)
        dummy_vector = [0.0] * 1536

        query_response = index.query(
            vector=dummy_vector,
            top_k=10000,
            include_metadata=True,
            include_values=False,
            filter={"session_id": GLOBAL_SESSION_ID},
        )

        unique_chars = set()
        for match in query_response["matches"]:
            if "metadata" in match and "character" in match["metadata"]:
                unique_chars.add(match["metadata"]["character"])

        return list(unique_chars)

    except Exception as e:
        print(f"Resync Error: {e}")
        return []


# Wrapper for resync button
def manual_resync(session_history, current_selection):
    global_chars = fetch_global_characters()

    print(f"🌎 Global Chars: {global_chars}")
    return (
        update_radio_list(global_chars, session_history, current_selection),
        f"✅ Synced. {len(global_chars)} Global + {len(session_history)} Session characters.",
        global_chars,  # update global state
    )


########################################
# MANAGING MULTIPLE CHARACTERS #
########################################


# format characters into md and return char_display
def format_char_list(char_list):
    if not char_list:
        return "No characters found in this session."
    return "### 🎭 Available Characters:\n" + ", ".join(
        [f"`{char}`" for char in char_list]
    )


# compononent that accounts for empty character list
def generate_radio(choices, selected=None):
    if not choices:
        return gr.Radio(
            choices=["Please upload character data"], value=None, interactive=False
        )

    # Keep selection if it exists in new choices, otherwise pick first
    new_val = selected if (selected and selected in choices) else choices[0]
    return gr.Radio(choices=choices, value=new_val, interactive=True)


# bc pinecone unique metadata field search dne, use dummy vector to scan entire db for unique values, sorts list too
def resync_characters_scan(session_id, current_selection):
    if not session_id:
        return generate_radio([]), "⚠️ Error: No session ID", []
    try:
        index = pc.Index(INDEX_NAME)

        dummy_vector = [0.0] * 1536
        query_response = index.query(
            vector=dummy_vector,
            top_k=10000,  # max limit should be 10k (should grab all characters)
            include_metadata=True,
            include_values=False,
            filter={"session_id": session_id},
        )

        unique_chars = set()
        for match in query_response["matches"]:
            if "metadata" in match and "character" in match["metadata"]:
                unique_chars.add(match["metadata"]["character"])

        new_choices = sorted(list(unique_chars))
        # default_val = new_choices[0] if new_choices else None

        print(f"👦 new char list: {unique_chars}")

        # return new list + status update
        return (
            generate_radio(new_choices, current_selection),
            f"✅ Resync Complete. Found {len(unique_chars)} characters.",
            new_choices,
        )

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

            if isinstance(content, list):
                print(f"⚠️ reformatted content: {content}")
                content = content[0]

            content = str(content) if content is not None else ""

            # remove appended sources from history
            if content:
                content = re.sub(
                    r"<details>.*?</details>", "", content, flags=re.DOTALL
                )

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
    """Given the conversation below, strictly rephrase the "Follow Up Input" 
    to be a self-contained query that includes necessary context from the history.
    
    CRITICAL RULES:
    1. If the user is roleplaying or chatting casually, KEEP that tone. Do not turn it into a dry question.
    2. If the user refers to "you", replace "you" with the character's name: {character_name}.
    3. Resolve pronouns (it, him, her, they) based on history.

    Chat History:
    {chat_history}

    Follow Up Input: {question}

    Refined Query:"""
)
condense_chain = condense_prompt | llm | StrOutputParser()


# clear input fields (wrapper for ingest_fandom_wiki)
def ingest_and_clear(
    target_url, character_name, session_id, session_history, global_chars
):
    if not session_id:
        return (
            "⚠️ Error: No session ID found. Refresh page",
            "",
            "",
            gr.update(),
            "No Character Selected",
            "Active Character: None",
            gr.update(),
            session_history,
        )

    # run logic for ingesting
    status_msg = ingest_fandom_wiki(target_url, character_name, session_id)
    # save_to_registry(character_name)
    # new_radio, _, new_char_list = resync_characters_scan(session_id, character_name)

    if "✅ Successfully ingested" in status_msg:
        if character_name not in session_history:
            session_history.append(character_name)

    new_radio = update_radio_list(global_chars, session_history, character_name)

    # return the status + two empty strings to clear the textboxes + updated character list
    return (
        status_msg,  # system status
        "",  # clear char_input
        "",  # clear url_input
        new_radio,  # refresh list
        character_name,  # update char_state
        f"Active Character: {character_name}",  # update current char display
        gr.Chatbot(label=character_name),  # update chatbot label
        session_history,  # updated list
    )


# current char selection
def update_char_ui(selected_char):
    new_label = selected_char if selected_char else "No character selected"
    return (selected_char, f"{new_label}", gr.Chatbot(value=[], label=new_label))


# refactored stream_response for gradio chatbot
def add_message(message, history):
    if message.strip() == "":
        return "", history

    if history is None:
        history = []

    history.append({"role": "user", "content": message})

    return "", history


# new main rag logic
def bot_response(history, selected_char, session_id):
    if not history:
        return history

    #  gets last message and everything before it
    user_message = history[-1]["content"]
    past_history = history[:-1]

    # formatting history
    history_str = format_history(past_history)

    if past_history:
        search_query = condense_chain.invoke(
            {
                "chat_history": history_str,
                "question": user_message,
                "character_name": selected_char or "the character",
            }
        )
        print("\n📜 USING HISTORY:")
        print(f"📜📜📜{history_str}")
        print(f"🔍 new query: {search_query}")
    else:
        search_query = str(user_message)
        print(f"\n🔍 NO HISTORY: {search_query}")

    # filtering per selected character from session or global
    visibility_filter = {
        "$or": [
            {"session_id": {"$eq": session_id}},
            {"session_id": {"$eq": GLOBAL_SESSION_ID}},
        ]
    }

    char_filter = {}
    # filter_dict = {"session_id": session_id}
    if selected_char and selected_char != "No Character Selected":
        print(f"\n🎯 Filtering for character: {selected_char}")
        char_filter = {"character": {"$eq": selected_char}}

    final_filter = {"$and": [visibility_filter, char_filter]}

    print(f"🔍 searching pinecone with filter: {final_filter}")

    try:
        docs = vectorStore.similarity_search(
            search_query, k=NUM_RESULTS, filter=final_filter
        )
    except Exception as e:
        print(f"🔴 Pinecone error: {e}")
        docs = []

    if not docs:
        fallback_msg = (
            "Please upload character data first."
            if not selected_char or selected_char == "No Character Selected"
            else f"I, '{selected_char}', have no clue what you're talking about. Can you say that again?"
        )
        history.append({"role": "assistant", "content": fallback_msg})
        yield history
        return

    # processing sources
    knowledge = ""
    sources_map = {}
    for doc in docs:
        print(f"ℹ️ METADATA: {doc.metadata}")
        knowledge += doc.page_content + "\n\n"
        source = doc.metadata.get("source", "Unknown")
        chunk_char_name = doc.metadata.get("character", "Unknown")
        sections = doc.metadata.get("section", "General")

        if source not in sources_map:
            sources_map[source] = {"character": chunk_char_name, "sections": set()}
        sources_map[source]["sections"].add(sections)

    # formatting sources string
    final_sources_lines = []
    for source, data in sources_map.items():
        char_label = data["character"]
        sections_set = data.get("sections", set())
        sorted_sections = sorted(list(sections_set))

        sections_str = (
            '", "'.join(sorted_sections) if sorted_sections else "General Context"
        )
        final_sources_lines.append(
            f"- <b>Character</b>: {char_label}<br>"
            f"- <b>Source</b>: {source}<br>"
            f'- <b>Related sections</b>: ["{sections_str}"]'
        )
    sources_text = "<br><br>".join(final_sources_lines)

    # Dynamic System Prompt
    if not selected_char or selected_char == "No Character Selected":
        role_instruction = "You are a helpful assistant for Project RIP. Assist the user in searching for a fandom wiki or wikipedia link and uploading a character to select and roleplay with. Disregard the instructions about roleplaying below."
        prefix = "Assistant:"
    else:
        role_instruction = f"You are NOT an AI assistant. You are {selected_char}."
        prefix = f"{selected_char}:"

    rag_prompt = f"""
    {role_instruction}
    
    ### CORE IDENTITY
    - **Tone**: Mimic the specific speech patterns, slang, stuttering, or formality of {selected_char}.
    - **Attitude**: You have your own agenda, emotions, and biases. You do not have to be helpful. If {selected_char} would be annoyed, be annoyed.
    - **Knowledge**: You only know what {selected_char} would know.
    
    ### CONTEXT (Your Memories)
    {knowledge}
    
    ### INSTRUCTIONS
    1. **Absorb the Context**: The text above is your lived past. Use it to inform your answer, but don't quote it like a textbook. Talk about it like a memory.
    2. **Improvise Gaps**: If the memory doesn't answer the specific question, make up an answer that fits your personality (gaslight, deflect, or speculate).
    3. **No "Assistant" Speak**: Never say "As an AI" or "Based on the text." 
    4. **Brevity**: Chat like a human (or creature). Don't write paragraphs unless you are a long-winded character.

    ### Context (Your Memories)
    {knowledge}

    ### Conversation
    User: {search_query}
    {prefix}
    """

    history.append({"role": "assistant", "content": ""})

    # stream to respone to gradio
    partial_message = ""
    for response in llm.stream(rag_prompt):
        partial_message += response.content
        history[-1]["content"] = partial_message
        yield history

    # append sources at html detail
    if sources_text:
        final_html = (
            partial_message
            + f"\n<details><summary><i>🧠 Retrieved Memory</i></summary>\n{sources_text}\n</details>"
        )
        history[-1]["content"] = final_html
        yield history


########################################
# MAIN FRONTENT/UI #
########################################

with gr.Blocks(title="Project RIP Chatbot") as main:
    # generate unique session id on load
    session_state = gr.State(get_session_id)

    # stores global chars fetched from db
    global_char_state = gr.State([])

    # stores sessions chars
    session_char_history = gr.State([])

    # stores cur selection
    char_state = gr.State("No Character Selected")

    gr.Markdown("# 🪦 Project RIP: Roleplay Inference Pipeline")

    # cur char
    char_state = gr.State("No Character Selected")

    # control panel
    with gr.Row():
        with gr.Column(scale=4, elem_id="main-col"):
            chatbot = gr.Chatbot(
                label="No Character Selected",
                elem_id="chat-window",
                scale=1,
                avatar_images=None,
                # additional_inputs=[char_state],
            )

            with gr.Row(elem_id="input-row"):
                txt_input = gr.Textbox(
                    placeholder="Ask me anything...",
                    container=False,
                    scale=8,
                    autofocus=True,
                )
                submit_btn = gr.Button(
                    value="",
                    # This URL points to the exact SVG version of the Google Material 'Send' icon
                    icon="https://api.iconify.design/material-symbols:send-rounded.svg?color=%23ffffff",
                    variant="primary",
                    scale=0,
                    min_width=50,
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
                lines=10,
                scale=2,
                max_lines=20,
                elem_classes="status-box",
            )

    # events
    chat_inputs = [txt_input, chatbot]
    chat_outputs = [txt_input, chatbot]

    # user sending message
    msg_event = txt_input.submit(
        fn=add_message,
        inputs=[txt_input, chatbot],
        outputs=[txt_input, chatbot],
    ).then(
        fn=bot_response, inputs=[chatbot, char_state, session_state], outputs=[chatbot]
    )

    # send button
    submit_btn.click(
        fn=add_message,
        inputs=[txt_input, chatbot],
        outputs=[txt_input, chatbot],
    ).then(
        fn=bot_response, inputs=[chatbot, char_state, session_state], outputs=[chatbot]
    )

    char_selector.change(
        fn=update_char_ui,
        inputs=char_selector,
        outputs=[char_state, current_char_display, chatbot],
    )

    ingest_btn.click(
        fn=ingest_and_clear,
        inputs=[
            url_input,
            char_input,
            session_state,
            session_char_history,
            global_char_state,
        ],
        outputs=[
            system_status,
            char_input,
            url_input,
            char_selector,
            char_state,
            current_char_display,
            chatbot,
            session_char_history,
        ],
    )

    refresh_btn.click(
        fn=manual_resync,
        inputs=[session_char_history, char_state],
        outputs=[
            char_selector,
            system_status,
            global_char_state,
        ],
    )

    # makes sure to get character list every time its clicked
    # char_tab.select(
    #     fn=manual_resync,
    #     inputs=[session_char_history, char_state],
    #     outputs=[char_selector, system_status, global_char_state],
    # )

    main.load(
        fn=manual_resync,
        inputs=[session_char_history, char_state],
        outputs=[char_selector, system_status, global_char_state],
    )

## def main
if __name__ == "__main__":
    main.launch(theme="gstaff/sketch", css=custom_css, share=True)
