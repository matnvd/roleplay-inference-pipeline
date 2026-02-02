import os
import re
import uuid

import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from ingest import ingest_fandom_wiki
from notifier import send_traffic_notification

#########################################
# CONFIGURATIONS #
########################################

load_dotenv()
INDEX_NAME = "project-rip"
API_KEY = os.getenv("PINECONE_API_KEY")
GLOBAL_SESSION_ID = "demo_roster"
GLOBAL_CHAR_CACHE = []  # for hugging face storage
NUM_RESULTS = 8  # increase if want more chunks

ACCESS_CODES = {
    os.getenv("ADMIN_PASSWORD"): "Admin",
    os.getenv("RESUME_PASSWORD"): "Resume",
    os.getenv("FRIENDS_PASSWORD"): "Friends",
    os.getenv("FAMILY_PASSWORD"): "Family",
    os.getenv("PASSWORD1"): "Source1",
    os.getenv("PASSWORD2"): "Source2",
    os.getenv("PASSWORD3"): "Source3",
    os.getenv("PASSWORD4"): "Source4",
    os.getenv("PASSWORD5"): "Source5",
    os.getenv("PASSWORD6"): "Source6",
    os.getenv("PASSWORD7"): "Source7",
    os.getenv("PASSWORD8"): "Source8",
    os.getenv("PASSWORD9"): "Source9",
    os.getenv("PASSWORD10"): "Source10",
    os.getenv("PASSWORD11"): "Source11",
    os.getenv("PASSWORD12"): "Source12",
    os.getenv("PASSWORD13"): "Source13",
    os.getenv("PASSWORD14"): "Source14",
    os.getenv("PASSWORD15"): "Source15",
    os.getenv("PASSWORD16"): "Source16",
    os.getenv("PASSWORD17"): "Source17",
    os.getenv("PASSWORD18"): "Source18",
    os.getenv("PASSWORD19"): "Source19",
    os.getenv("PASSWORD20"): "Source20",
    os.getenv("PASSWORD21"): "Source21",
    os.getenv("PASSWORD22"): "Source22",
    os.getenv("PASSWORD23"): "Source23",
    os.getenv("PASSWORD24"): "Source24",
    os.getenv("PASSWORD25"): "Source25",
    os.getenv("PASSWORD26"): "Source26",
    os.getenv("PASSWORD27"): "Source27",
    os.getenv("PASSWORD28"): "Source28",
    os.getenv("PASSWORD29"): "Source29",
    os.getenv("PASSWORD30"): "Source30",
    os.getenv("PASSWORD31"): "Source31",
    os.getenv("PASSWORD32"): "Source32",
    os.getenv("PASSWORD33"): "Source33",
    os.getenv("PASSWORD34"): "Source34",
    os.getenv("PASSWORD35"): "Source35",
    os.getenv("PASSWORD36"): "Source36",
    os.getenv("PASSWORD37"): "Source37",
    os.getenv("PASSWORD38"): "Source38",
    os.getenv("PASSWORD39"): "Source39",
    os.getenv("PASSWORD40"): "Source40",
    os.getenv("PASSWORD41"): "Source41",
    os.getenv("PASSWORD42"): "Source42",
    os.getenv("PASSWORD43"): "Source43",
    os.getenv("PASSWORD44"): "Source44",
    os.getenv("PASSWORD45"): "Source45",
    os.getenv("PASSWORD46"): "Source46",
    os.getenv("PASSWORD47"): "Source47",
    os.getenv("PASSWORD48"): "Source48",
    os.getenv("PASSWORD49"): "Source49",
}

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# upgrade model in future? use better one? use fine-tuned one?
llm = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")

vectorStore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings_model,
)
pc = Pinecone(api_key=API_KEY)

#########################################
# CUSTOM CSS #
########################################
custom_css = """
/* ----------------------------------------------------------------------
    CHAT INTERFACE & BUBBLES
---------------------------------------------------------------------- */
#chat-window {
    background: transparent !important;
    background-color: transparent !important;
    # border: none !important;
    height: 60vh !important;
}

#chat-window .block,
#chat-window .wrap,
#chat-window .bubble-wrap {
    background: transparent !important;
    border: none !important;
}

/* spacing btwn bubbles */
#chat-window .message-wrap {
    gap: 15px;
}

/* Avatar Styling */
#chat-window .avatar-container img {
    border: none !important;
    border-width: 0 !important;
    box-shadow: none !important;
    background-color: transparent !important;
    padding: 0px !important
}

/* Message Bubbles Base */
#chat-window .message {
    # background-color: rgba(30, 30, 30, 0.4) !important;
    # backdrop-filter: blur(8px) !important;
    # border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    border-radius: 15px !important;
    color: white !important
}

/* User Bubble Specifics */
#chat-window .message.user {
    # border-bottom-right-radius: 2px !important;
    background-color: rgba(60, 60, 80, 0.5) !important;
}

/* Bot Bubble Specifics */
#chat-window .message.bot {
    # border-bottom-left-radius: 2px !important;
    background-color: rgba(40, 40, 40, 0.5) !important;
}

/* ==========================================================================
    DESKTOP RULES (min-width: 768px)
========================================================================== */
@media (min-width: 768px) {

    /* ----------------------------------------------------------------------
        GLOBAL LAYOUT & CONTAINERS
    ---------------------------------------------------------------------- */
    .gradio-container {
        min-height: 100vh !important;
        height: auto !important;
        overflow-y: auto !important; /* Enable scrolling */
    }

    /* MAIN APP WRAPPER */
    #main-app {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        padding: 0 !important;
        
        max-width: 1200px !important;
        margin-left: auto !important;
        margin-right: auto !important;

        overflow: visible !important;
        height: auto !important;
        # padding-top: 0px !important;
    }

    #main-app > .block,
    #main-app > .form,
    #main-app > .wrap {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    /* HEADER ROW (Title + Logout) */
    
    /* Fix header scrollbars */
    .header-row, 
    .header-row > .col, 
    .header-row .block {
        overflow: visible !important;
    }
    
    .header-row {
        align-items: center !important;
        margin-bottom: 10px !important;
    }

    .header-row h1 {
        margin: 0 !important;
    }

    /* MAIN COLUMN FLEX CONTAINER */
    #main-col {
        # padding: 0 !important;
        height: auto !important;
        display: flex !important;
        flex-direction: column !important;
    }

    #main-col .prose {
        font-size: 16px !important;
        line-height: 1.5 !important;
    }


    /* ----------------------------------------------------------------------
        LOGIN SCREEN
    ---------------------------------------------------------------------- */
    #login-screen {
        # height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        padding-top: 20px !important;
    }

    #login-screen .block,
    #login-screen .form {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    #login-screen > .row {
        width: 100% !important;
        max-width: 450px !important;
    }
    
    /* Login Text Styling */
    #login-screen h1, #login-screen p {
        # color: white !important;
        text-align: center !important;
    }

    /* Login Input Row */
    #login-row {
        align-items: center !important;
        gap: 10px !important;
    }

    #login-row .block {
        background: transparent !important;
    }

    #login-row button {
        height: 100% !important;
        min-height: 42px !important;
    }

    #login-row textarea,
    #login-row input {
        background-color: rgba(82, 82, 82, 0.4) !important;
        # color: white !important;
        # border: 1px solid rgba(255, 255, 255, 0.7) !important;
        min-height: 46px !important;
    }

    #login-row input:focus {
        border-color: var(--color-accent) !important;
        background-color: rgba(82, 82, 82, 0.3) !important;
    }

    /* login video */
    #login-demo-video {
        max-width: 800px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        margin-top: 20px !important;
        margin-bottom: 20px !important;
        display: block !important;
    }

    #login-demo-video > .prose {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }

    /* ----------------------------------------------------------------------
        INPUT ROW & CONTROLS
    ---------------------------------------------------------------------- */
    #input-row {
        align-items: stretch !important;
        gap: 8px !important;
    }
    
    #chat-input {
        overflow: visible !important;
        border-radius: 12px !important;
    }

    #chat-input textarea {
        overflow-y: auto !important; 
        max-height: 150px !important;
        border-radius: 12px !important;
    }

    #send-btn {
        height: 100% !important;
        margin: 0 !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* General Input Field Styling (fixes max_lines bug) */
    .no-wrap textarea, .no-wrap input {
        white-space: nowrap !important;
        overflow-x: auto !important;
        background-color: var(--input-background-fill) !important;
    }

    /* Hide arrows in temp slider */
    input[type=number]::-webkit-inner-spin-button, 
    input[type=number]::-webkit-outer-spin-button { 
        -webkit-appearance: none !important; 
        margin: 0 !important; 
    }
    input[type=number] {
        -moz-appearance: textfield !important;
    }


    /* ----------------------------------------------------------------------
        SIDEBAR COMPONENTS
    ---------------------------------------------------------------------- */
    /* Tabs Expansion */
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

    /* Character List (Radio Group) */
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
        color: var(--body-text-color) !important;
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
    
    .char-radio-group input[type="radio"] {
        display: none !important; /* Hide default radio bubble */
    }

    /* System Status Box */
    .status-box {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        min-height: 0 !important;
    }

    .status-box textarea {
        height: 85px !important;
        overflow-y: auto !important;
    }
}

/* ==========================================================================
    MOBILE RULES (max-width: 767px)
   ========================================================================== */
@media (max-width: 767px) {
    #main-app {
        padding: 10px !important; /* Add small padding for mobile edges */
    }
    
    /* Fix chat window height on mobile so it doesn't take up the whole page */
    #chat-window {
        height: 500px !important; 
    }

    #login-demo-video iframe {
        width: 100% !important;
        height: auto !important;
        aspect-ratio: 16 / 10;
    }
}
"""

########################################
# MANAGING MULTIPLE SESSIONS #
########################################


def get_session_id():
    new_id = str(uuid.uuid4())
    print(f"\n\n🆕 NEW SESSION STARTED: {new_id}")
    return new_id


# scan pinecone for global characters (& their images) only
def fetch_global_characters():
    global GLOBAL_CHAR_CACHE
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
        image_map = {}

        for match in query_response["matches"]:
            if "metadata" in match:
                meta = match["metadata"]
                char_name = meta.get("character")
                img_url = meta.get("image_url")

                if char_name:
                    unique_chars.add(char_name)

                    if char_name not in image_map and img_url:
                        image_map[char_name] = img_url

        char_list = list(unique_chars)
        GLOBAL_CHAR_CACHE = char_list
        return char_list, image_map

    except Exception as e:
        print(f"Resync Error: {e}")
        return [], {}


# fetch immediately
fetch_global_characters()


# runs on page load, using cached global list
def on_app_load():
    chars, global_images = fetch_global_characters()

    new_radio = update_radio_list(chars, [], None)

    return (
        new_radio,
        f"🟢 Ready. Loaded {len(chars)} characters.",
        chars,
        global_images,
    )


# wrapper for resync button
def manual_resync(session_history, current_selection):
    global_chars, image_map = fetch_global_characters()

    print(f"🌎 Global Chars: {global_chars}")
    return (
        update_radio_list(global_chars, session_history, current_selection),
        f"✅ Synced. {len(global_chars)} Global + {len(session_history)} Session characters.",
        global_chars,  # update global state
        image_map,
    )


# updating char list
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

    # keep selection if valid, else pick first
    new_val = (
        selected if (selected and selected in combined_choices) else combined_choices[0]
    )

    return gr.Radio(choices=combined_choices, value=new_val, interactive=True)


# current char selection, saving old character data before loading new one
def save_and_switch_character(
    new_selection, old_selection, current_history, history_map, image_map
):
    # dont change if same selection
    if new_selection == old_selection:
        return (
            old_selection,
            f"Speaking With: {old_selection}",
            gr.update(),
            history_map,
        )

    # store chat history before leaving
    if old_selection and old_selection != "No Character Selected":
        history_map[old_selection] = current_history

    # retrieve history
    new_history = history_map.get(new_selection, [])

    # load image
    avatar_url = image_map.get(new_selection, None)

    new_label = new_selection if new_selection else "No Character Selected"

    return (
        new_selection,  # new char_state
        f"Speaking With: {new_label}",  # new current char display
        gr.Chatbot(  # new chatbot label and image
            value=new_history, label=new_label, avatar_images=(None, avatar_url)
        ),
        history_map,  # updated histories
    )


########################################
# CHATBOT - HISTORY/SOURCING #
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
    1. If the user refers to "you", replace "you" with the character's name: {character_name}.
    2. Resolve pronouns (it, him, her, they) based on history.
    3. **NO HALLUCINATION**: If the Follow Up Input is gibberish, keyboard smashing (e.g. "asdf"), or completely unrelated to the history, RETURN IT EXACTLY AS IS. Do not try to make sense of it.
    4. If the user is just saying "hello" or a short reaction, return it exactly as is.

    Chat History:
    {chat_history}

    Follow Up Input: {question}

    Refined Query:"""
)
condense_chain = condense_prompt | llm | StrOutputParser()


# clears chats to prevent previous character "typing"
def start_ingest(char_name):
    temp_label = f"Connecting you to {char_name or 'Unknown'}..."

    return (
        "⏳ Connecting to Wiki...",
        gr.Chatbot(value=[], label=temp_label, avatar_images=None),
        "Status: Uploading...",
    )


# clear input fields (wrapper for ingest_fandom_wiki)
def ingest_and_clear(
    target_url,
    character_name,
    session_id,
    session_history,
    global_chars,
    traffic_source,
    last_char,
    image_map,
    history_map,
    current_chat_history,
):
    # security check
    if not traffic_source or traffic_source == "Unknown":
        raise gr.Error("⛔ Unauthorized: Please log in first.")

    if not session_id:
        return (
            "⚠️ Error: No session ID found. Refresh page",
            "",
            "",
            gr.update(),
            "No Character Selected",
            "No Active Character",
            gr.update(),
            last_char,
            session_history,
            image_map,
            history_map,
        )

    if "wikipedia.org" not in target_url and "fandom.com" not in target_url:
        return (
            "⚠️ Error: Invalid URL. Only 'wikipedia.org' and 'fandom.com' links are supported.",  # system status
            character_name,  # don't clear name so user doesn't have to retype
            target_url,  # Don't clear URL so user can fix it
            gr.update(),  # No change to character list
            gr.update(),  # dont change current character
            gr.update(),  # dont change character display
            gr.update(),  # dont chant label
            gr.update(),  # last_char, shouldn't change
            session_history,
            image_map,
            history_map,
        )

    if last_char and last_char != "No Character Selected":
        history_map[last_char] = current_chat_history

    # run logic for ingesting
    status_msg, img_url = ingest_fandom_wiki(
        target_url, character_name, session_id, traffic_source
    )

    if "✅ Successfully ingested" in status_msg:
        # dynamically update characters
        if character_name not in session_history:
            session_history.append(character_name)

        if img_url:
            image_map[character_name] = img_url

        # start with empty history
        history_map[character_name] = []

    # update character list and get image
    new_radio = update_radio_list(global_chars, session_history, character_name)
    new_avatar = image_map.get(character_name, None)

    return (
        status_msg,  # system status
        "",  # clear char_input
        "",  # clear url_input
        new_radio,  # refresh list
        character_name,  # update char_state
        f"Speaking With: {character_name}",  # update current char display
        gr.Chatbot(
            value=[], label=character_name, avatar_images=(None, new_avatar)
        ),  # update chatbot label and pfp
        session_history,  # updated list of sesion characters
        image_map,  # updated list of character pfps
        history_map,  # updated list of character histories
    )


# have bot prompt user first
def trigger_greeting(selected_char, session_id, temperature, history):
    # skip if already history
    if history and len(history) > 0:
        yield history
        return

    # skip if no character is selected
    if not selected_char or selected_char == "No Character Selected":
        yield history
        return

    print(f"👋 Triggering greeting for: {selected_char}")

    # same filter from bot_response
    visibility_filter = {
        "$or": [
            {"session_id": {"$eq": session_id}},
            {"session_id": {"$eq": GLOBAL_SESSION_ID}},
        ]
    }
    char_filter = {"character": {"$eq": selected_char}}
    final_filter = {"$and": [visibility_filter, char_filter]}

    # search for "identity" or "intro" specifically for greeting prompt
    docs = vectorStore.similarity_search(
        "Who am I? Personality, introduction, and famous quotes.",
        k=5,  # brief search
        filter=final_filter,
    )

    knowledge = "\n".join([doc.page_content for doc in docs])

    greeting_prompt = f"""
    You are {selected_char}.
    
    ### YOUR CONTEXT (Memories)
    {knowledge}
    
    ### INSTRUCTION
    You have just encountered a new user.
    Generate a short, engaging opening line (1-2 sentences) to start the conversation.
    
    CRITICAL RULES:
    1. **Stay In Character**: If you are a villain, be arrogant. If you are a shy anime girl, stutter.
    2. **Prompt the User**: Give them a reason to reply (ask a question, make a demand, or comment on the surroundings).
    3. **NO AI Slop**: Do NOT say "How can I help you?" or "I am ready to chat."
    
    Start the conversation now:
    """

    # initialize history with one empty assistant message
    history = [{"role": "assistant", "content": ""}]

    dynamic_llm = llm.bind(temperature=temperature)

    partial_message = ""
    for response in dynamic_llm.stream(greeting_prompt):
        partial_message += response.content
        history[-1]["content"] = partial_message
        yield history


# refactored stream_response for gradio chatbot
def add_message(message, history, traffic_source, has_notified, session_id):
    # security check, if they bypassed the login screen, give them an error
    if not traffic_source or traffic_source == "Unknown":
        raise gr.Error("⛔ Unauthorized: Please log in first.")

    if message.strip() == "":
        return "", history

    if history is None:
        history = []

    # traffic sourcing
    if not has_notified:
        if traffic_source != "Admin":
            print(f"🔔 First message from: {traffic_source}")
            send_traffic_notification(traffic_source, session_id)
        else:
            print(f"🔔 Skipped traffic notification from: {traffic_source}")

        has_notified = True

    history.append({"role": "user", "content": message})

    return "", history, has_notified


# main rag chatbot logic
def bot_response(history, selected_char, session_id, temperature):
    if not history:
        return history

    #  gets last message and everything before it
    user_message = history[-1]["content"]
    past_history = history[:-1]

    # formatting history
    history_str = format_history(past_history)

    # look for prev chat history
    if past_history:
        # rephrase current query given history to make more sense
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
    if selected_char and selected_char != "No Character Selected":
        print(f"\n🔍 Filtering for character: {selected_char}")
        char_filter = {"character": {"$eq": selected_char}}

    final_filter = {"$and": [visibility_filter, char_filter]}

    print(f"🔍 searching Pinecone with filter: {final_filter}")

    # searching up RAG docs
    try:
        docs = vectorStore.similarity_search(
            search_query, k=NUM_RESULTS, filter=final_filter
        )
    except Exception as e:
        print(f"🔴 Pinecone error: {e}")
        docs = []

    # basically resort to general knowledge if none are found
    if not docs:
        print("🟡 No relevant docs found!")

    # processing sources
    knowledge = ""
    sources_map = {}
    for doc in docs:
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

    # temperature-based instructions
    if temperature <= 0.3:
        # low temp: more stabile and calm
        temp_guidance = (
            "You are calm, collected, and precise. "
            "Stick strictly to the facts in your Context. "
            "Minimize slang and emotional outbursts."
        )
    elif temperature >= 0.9:
        # high temp: more chotic and dramatic
        temp_guidance = (
            "You are erratic, emotional, and dramatic. "
            "Dial your personality traits up to 11. "
            "If the Context is boring, spice it up with your own opinions or wild theories. "
            "Don't be afraid to be rude, weird, or unhelpful if it fits your character."
        )
    else:
        # mid temp
        temp_guidance = (
            "Act naturally. "
            "Balance your specific personality quirks with the factual Context provided."
        )

    # dynamic system prompt (in case it errors out and no character selected)
    if not selected_char or selected_char == "No Character Selected":
        role_instruction = "You are a helpful assistant for Project RIP. Assist the user in searching for a fandom wiki or wikipedia link and uploading a character to select and roleplay with. Disregard the instructions about roleplaying below."
        prefix = "Assistant:"
    else:
        role_instruction = f"You are NOT an AI assistant. You are {selected_char}."
        prefix = f"{selected_char}:"

    # acc prompt, prompt engineering framework:
    rag_prompt = f"""
    {role_instruction}

    ### CURRENT MOOD SETTING
    {temp_guidance}
    
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

    ### Conversation
    User: {search_query}
    {prefix}
    """

    history.append({"role": "assistant", "content": ""})

    # stream to respone to gradio
    partial_message = ""

    # set temperature for this specific run
    dynamic_llm = llm.bind(temperature=temperature)

    print(f"🌡️ curTemp: {temperature}")
    for response in dynamic_llm.stream(rag_prompt):
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
# SECURITY #
########################################


# authentication function
def verify_login(password):
    if password in ACCESS_CODES:
        source_label = ACCESS_CODES[password]
        print(f"🔓 Traffic source: {source_label}")

        # hide main app and show login screen (can technically get around but functions still shouldnt work)
        return {
            login_col: gr.Column(visible=False),
            main_app_col: gr.Column(visible=True),
            traffic_source_state: source_label,
            login_error_msg: gr.Markdown("", visible=False),
        }
    else:
        print(f"🔒 Failed Login Attempt: {password}")
        return {
            login_col: gr.Column(visible=True),
            main_app_col: gr.Column(visible=False),
            traffic_source_state: "Unknown",
            login_error_msg: gr.Markdown("❌ Incorrect Access Code", visible=True),
        }


# For going back to login page
def logout():
    print("🔒 User logged out")
    return {
        login_col: gr.Column(visible=True),
        main_app_col: gr.Column(visible=False),
        traffic_source_state: "Unknown",
        pass_input: "",
        login_error_msg: gr.Markdown(visible=False),
    }


########################################
# MAIN FRONTEND/UI #
########################################

with gr.Blocks(title="Project RIP Chatbot") as main:
    # generate unique session id on load
    session_state = gr.State(get_session_id)
    login_tracker_state = gr.State(False)
    traffic_source_state = gr.State("Unknown")

    # stores global chars fetched from db
    global_char_state = gr.State([])

    # stores sessions chars
    session_char_history = gr.State([])

    # stores cur selection
    char_state = gr.State("No Character Selected")
    char_images_state = gr.State({})
    chat_history_map = gr.State({})

    # custom login screen (instead of default auth)
    with gr.Column(elem_id="login-screen", visible=True) as login_col:
        gr.Markdown(
            """
            # 🔒 Project RIP Access
            Please enter the access code to continue <br>(should be next to the link on resume).
            """,
            elem_classes=["prose"],
        )

        with gr.Row(elem_id="login-row"):
            pass_input = gr.Textbox(
                label="Access Code",
                type="password",
                placeholder="Enter password...",
                show_label=False,
                scale=4,
                autofocus=True,
            )
            login_btn = gr.Button("Enter", variant="primary", scale=1)

        gr.HTML(
            """
            <div style="text-align: center; margin-top: 20px; font-size: 1.1em;">
                <a href="https://github.com/matnvd/roleplay-inference-pipeline" target="_blank" style="text-decoration: none; color: var(--color-accent); margin-right: 15px;">
                    📂 GitHub Repository
                </a>

                <span style="color: gray;">|</span>
                
                <a href="mailto:mathiasnvd07@gmail.com?subject=Project%20RIP%20Bug%20Report" style="text-decoration: none; color: var(--color-accent); margin-left: 15px;">
                    🐛 Report Bug
                </a>
            </div>
            """
        )

        login_error_msg = gr.Markdown("", visible=False)

        embed_html = """
        <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
            <iframe width="644" height="400" 
                src="https://www.youtube.com/embed/2Y4AmMF2oZM?si=hF2Go23ietXc0V5t" 
                title="YouTube video player" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen 
                style="border-radius: 12px;">
            </iframe>
        </div>
        """

        gr.HTML(embed_html, elem_id="login-demo-video")

    # main app col
    with gr.Column(elem_id="main-app", visible=False) as main_app_col:
        with gr.Row(elem_classes=["header-row"]):
            with gr.Column(scale=8):
                gr.Markdown("# 🪦 Project RIP: Roleplay Inference Pipeline")
            logout_btn = gr.Button(
                "🚪 Log Out", size="sm", variant="secondary", scale=1, min_width=100
            )

        # control panel
        with gr.Row():
            with gr.Column(scale=4, elem_id="main-col"):
                chatbot = gr.Chatbot(
                    label="No Character Selected",
                    elem_id="chat-window",
                    scale=1,
                    avatar_images=None,
                )

                with gr.Row(elem_id="input-row"):
                    txt_input = gr.Textbox(
                        placeholder="Ask me anything...",
                        container=False,
                        scale=8,
                        autofocus=True,
                        elem_id="chat-input",
                    )
                    submit_btn = gr.Button(
                        value="",
                        # svg version of google fonts 'send' icon
                        icon="https://api.iconify.design/material-symbols:send-rounded.svg?color=%23ffffff",
                        variant="primary",
                        scale=0,
                        min_width=50,
                    )

                with gr.Row():
                    current_char_display = gr.Textbox(
                        label="Active Character",
                        value="No Character Available",
                        interactive=False,
                        lines=1,
                        elem_id="active-char-box",
                        scale=3,
                    )
                    temp_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.2,  # anything more breaks it
                        value=0.6,
                        step=0.1,
                        label="Temperature",
                        interactive=True,
                        scale=1,
                    )
            with gr.Column(scale=2):
                with gr.Tabs(elem_classes=["expand-tabs"]):
                    # tab 1: initial uploading character section
                    with gr.Tab("🚀 Upload Context", scale=1) as upload_tab:
                        gr.Markdown("""
                        **How to use:**
                        1. Upload your **own** character (or **choose** an existing one!).
                        1. Paste a **Wikipedia** or **Fandom Wiki** URL.
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

                        ingest_btn = gr.Button(
                            "🚀 Upload Data", variant="primary", scale=1
                        )

                    # tab 2: character selection
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

                # ingestion status + general status updates
                system_status = gr.Textbox(
                    label="System Status",
                    value="🟢 Ready 🟢",
                    interactive=False,
                    lines=8,
                    scale=2,
                    max_lines=10,
                    elem_classes="status-box",
                )

    js_focus = "() => document.querySelector('#chat-input textarea').focus()"

    # events
    login_targets = [login_col, main_app_col, traffic_source_state, login_error_msg]

    login_btn.click(fn=verify_login, inputs=pass_input, outputs=login_targets)
    pass_input.submit(fn=verify_login, inputs=pass_input, outputs=login_targets)

    # typing into text box
    chat_inputs = [
        txt_input,
        chatbot,
        traffic_source_state,
        login_tracker_state,
        session_state,
    ]
    chat_outputs = [txt_input, chatbot]

    # user sending message (via enter)
    msg_event = txt_input.submit(
        fn=add_message,
        inputs=chat_inputs,
        outputs=[txt_input, chatbot, login_tracker_state],
    ).then(
        fn=bot_response,
        inputs=[chatbot, char_state, session_state, temp_slider],
        outputs=[chatbot],
    )

    # sending message (via send button)
    submit_btn.click(
        fn=add_message,
        inputs=chat_inputs,
        outputs=[txt_input, chatbot, login_tracker_state],
    ).then(
        fn=bot_response,
        inputs=[chatbot, char_state, session_state, temp_slider],
        outputs=[chatbot],
    )

    # any triggering of changing character, then trigger greeting
    char_selector.change(
        fn=save_and_switch_character,
        inputs=[
            char_selector,
            char_state,
            chatbot,
            chat_history_map,
            char_images_state,
        ],
        outputs=[char_state, current_char_display, chatbot, chat_history_map],
    ).then(
        fn=trigger_greeting,
        inputs=[char_state, session_state, temp_slider, chatbot],
        outputs=[chatbot],
    )

    # upbloading data (start_ingest is to clear chat window frame while transitioning)
    ingest_btn.click(
        fn=start_ingest,
        inputs=[char_input],
        outputs=[system_status, chatbot, current_char_display],
    ).then(
        fn=ingest_and_clear,
        inputs=[
            url_input,
            char_input,
            session_state,
            session_char_history,
            global_char_state,
            traffic_source_state,
            char_state,
            char_images_state,
            chat_history_map,
            chatbot,
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
            char_images_state,
            chat_history_map,
        ],
    ).then(None, None, None, js=js_focus)

    # other way to ingest, by pressing enter (presumably entering url second)
    url_input.submit(
        fn=start_ingest,
        inputs=[char_input],
        outputs=[system_status, chatbot, current_char_display],
    ).then(
        fn=ingest_and_clear,
        inputs=[
            url_input,
            char_input,
            session_state,
            session_char_history,
            global_char_state,
            traffic_source_state,
            char_state,
            char_images_state,
            chat_history_map,
            chatbot,
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
            char_images_state,
            chat_history_map,
        ],
    ).then(None, None, None, js=js_focus)

    # manual refresh
    refresh_btn.click(
        fn=manual_resync,
        inputs=[session_char_history, char_state],
        outputs=[char_selector, system_status, global_char_state, char_images_state],
    )

    # logout button
    logout_btn.click(
        fn=logout,
        inputs=None,
        outputs=[
            login_col,
            main_app_col,
            traffic_source_state,
            pass_input,
            login_error_msg,
        ],
    )

    # main function on app load
    main.load(
        fn=on_app_load,
        inputs=None,
        outputs=[char_selector, system_status, global_char_state, char_images_state],
    )

# def main
if __name__ == "__main__":
    main.launch(theme="gstaff/sketch", css=custom_css, share=True)
