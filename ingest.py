import os
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

load_dotenv()

# The specific Wiki page we're ripping data from
INDEX_NAME = "project-rip"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


# for notifying pinecone db updates to avoid overusage
def send_upload_notification(character_name, source_url, session_id):
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ No Discord Webhook URL found in .env, skipping notification.")
        return

    payload = {
        "username": "Project RIP Ingestion Bot",
        # "avatar_url": "https://cdn-icons-png.flaticon.com/512/2040/2040504.png",
        "embeds": [
            {
                "title": "🚀 New Character Uploaded!",
                "color": 5763719,  # Green color code
                "fields": [
                    {"name": "🎭 Character", "value": character_name, "inline": True},
                    {
                        "name": "🔗 Source",
                        "value": f"[Wiki Link]({source_url})",
                        "inline": True,
                    },
                    {"name": "Session_ID", "value": session_id, "inline": True},
                ],
                "footer": {"text": "Project RIP Vector Database"},
            }
        ],
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print("🔔 Character upload Discord notification sent!")
    except Exception as e:
        print(f"❌ Failed to send upload notification: {e}")


# for notifying ab diff trafic sources
def send_traffic_notification(traffic_source, session_id):
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ No Discord Webhook URL found in .env, skipping notification.")
        return

    payload = {
        "username": "Project RIP Ingestion Bot",
        # "avatar_url": "https://cdn-icons-png.flaticon.com/512/2040/2040504.png",
        "embeds": [
            {
                "title": "🔓 New Login Alert!",
                "color": 15548997,  # red
                "fields": [
                    {
                        "name": "🌐 Traffic Source",
                        "value": traffic_source,
                        "inline": True,
                    },
                    {"name": "Session_ID", "value": session_id, "inline": True},
                ],
                "footer": {"text": "Project RIP Vector Database"},
            }
        ],
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print("🔔 Traffic source Discord notification sent!")
    except Exception as e:
        print(f"❌ Failed to send traffic notification: {e}")


# CONTENT COLLECTION
# web scrapes and cleans data to inject into db
# Fetches the raw HTML and isolates the main article text.
def rip_wiki_content(url):
    print(f"⚡ connecting to {url}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }  # Fakes a real browser so the Wiki doesn't block us

    try:
        response = requests.get(url, headers=headers)
        # print(f"response.text: {response.text}")
        response.raise_for_status()  # Crashes nicely if link is dead (404)
    except Exception as e:
        print(f"❌ Error fetching page: {e}")
        return None, None

    soup = BeautifulSoup(response.text, "html.parser")

    # print(f"soup.title: {soup.title.string if soup.title else 'No Title'}")
    # FINDING CONTENT
    # On Fandom and wikipedia, article usually inside <div class="mw-parser-output">.
    # ignore sidebars, ads, and footers.
    content_div = soup.find("div", {"class": "mw-parser-output"})
    if not content_div:
        # Fallback for weird Fandom layouts
        print("    🟡 weird fandom layout. searching mw-content-text.")
        content_div = soup.find("div", {"id": "mw-content-text"})

    if not content_div:
        print(
            "❌ Could not find main content div. Page structure is unknown, try uploading a wiki page."
        )
        return None, None

    # DATA CLEANING
    # We only want paragraphs <p>, lists <ul>, and headers <h>
    clean_text = ""
    section_map = []  # storing: {'start':0, 'end':500, 'header': 'Intro'}

    current_header = "Introduction"
    current_idx = 0

    tags_to_scrape = ["p", "h1", "h2", "h3", "h4", "ul"]  # add more later if needed

    for element in content_div.find_all(tags_to_scrape, recursive=False):
        tag = element.name
        text = element.get_text(" ", strip=True)

        if not text or len(text) < 3:
            continue

        # add header info in front of text
        if tag in ["h2", "h3"]:  # add h4?
            current_header = text
            text_entry = f"\n\n== {text} ==\n"
            clean_text += text_entry
            current_idx += len(text_entry)
        else:
            text_entry = text + "\n\n"
            start = current_idx
            end = current_idx + len(text_entry)

            section_map.append({"start": start, "end": end, "header": current_header})

        clean_text += text_entry
        current_idx = end

    print(
        f"✅ Successfully extracted {len(clean_text)} characters of raw text and mapped {len(section_map)} sections."
    )

    if not clean_text:
        print("⚠️ Content div found but no extracted text")

    return clean_text, section_map


# CHUNKING
# nlp pre-processing: splits long txt into chunks for vector db
def chunk_text(raw_text, section_map, source_url, character_name, session_id):
    print("🔪 Chunking text...")

    # splits Paragraph by (\n\n) first, then Sentence (.), then Word.
    # This prevents cutting a sentence in half, which confuses the AI.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # ~200-300 words per chunk, longer better for narrative context?
        chunk_overlap=400,  # Overlap ensures context flows across chunks
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = text_splitter.create_documents([raw_text])

    search_cursor = 0

    for chunk in chunks:
        content = chunk.page_content
        start_idx = raw_text.find(content, search_cursor)
        end_idx = start_idx + len(content)

        # find dominant section for each chunk
        if start_idx == -1:
            dominant_section = "Unknown"
        else:
            search_cursor = start_idx

            max_overlap = 0
            dominant_section = "Unknown"

            for entry in section_map:
                overlap_start = max(start_idx, entry["start"])
                overlap_end = min(end_idx, entry["end"])

                overlap_length = max(0, overlap_end - overlap_start)

                if overlap_length > max_overlap:
                    max_overlap = overlap_length
                    dominant_section = entry["header"]

        chunk.metadata = {
            "character": character_name,
            "source": source_url,
            "section": dominant_section,
            "session_id": session_id,  # change to "demo_roster" to upload global characters; in future, would need to make this id secret
        }

        # # find all sections that touch this chunk
        # found_sections = []
        # for entry in section_map:
        #     if entry["start"] < end_idx and entry["end"] > start_idx:
        #         found_sections.append(entry["header"])

        # chunk_sections = list(dict.fromkeys(found_sections))

        # if not chunk_sections:
        #     chunk_sections = ["Unknown"]

        # chunk.metadata = {"source": source_url, "section": chunk_sections}

    print(f"✅ Split text into {len(chunks)} vector-ready chunks.")
    return chunks


# caching, don't do unnecessary embeddings if already ripped that url
def check_if_url_exists(url, session_id):
    print(f"🔍 Checking if {url} is already in the database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorStore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    dummy_vector = [0.0] * 1536

    try:
        results = vectorStore.similarity_search_by_vector_with_score(
            embedding=dummy_vector,
            k=1,
            filter={"source": url, "session_id": session_id},
        )

        if len(results) > 0:
            print(f"    ✅ Found existing match! (Score: {results[0][1]})")
            return True

        print("    🟡 No match found in DB.")
        return False
    except Exception as e:
        print(f"🔴 Error checking URL existence: {e}")
        return False


# embed and send chunks to pinecone
def ingest_to_pinecone(chunks):
    # once working look at mteb leaderboard and change model? llama?
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorStore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    print(f"🚀 Uploading {len(chunks)} chunks to Pinecone index '{INDEX_NAME}'...")

    try:
        # # uploads to pinecone (sends chunks to openai to get vectors and forwards to pinecone)
        vectorStore.add_documents(documents=chunks)
        print("✅ Success! Data is now live in the Vector Database.")
    except Exception as e:
        print(f"❌ Upload failed: {e}")


# MAIN (exported) FUNCTION
def ingest_fandom_wiki(url, character_name, session_id, traffic_source):
    if not url:
        return "⚠️ No URL provided."

    if not session_id:
        return "⚠️ Error: No Session ID provided."

    if not character_name:
        character_name = "Unknown Character"  # add counter for num of characters

    parsed = urlparse(url)
    clean_url = urlunparse(parsed._replace(fragment="")).rstrip("/")

    # Debug: Show user what happened
    if url != clean_url:
        print(f"🧹 Normalized URL: '{url}' -> '{clean_url}'")

    if check_if_url_exists(clean_url, session_id):
        return f"⚠️ Skipping ingestion. '{clean_url}' is already in the DB."

    print(
        f"🔗 New URL detected for character '{character_name}': {clean_url} (Session: {session_id})"
    )

    # run extractor
    try:
        raw_data, map_data = rip_wiki_content(clean_url)
    except Exception:
        return "❌ Failed to scrape content."

    if not raw_data:
        return "❌ Failed to scrape content, no data available"

    # run transformer
    final_chunks = chunk_text(raw_data, map_data, clean_url, character_name, session_id)
    if not final_chunks:
        return "❌ Failed to create chunks"

    print("\n--- METADATA (preview): ---")
    for i in range(min(5, len(final_chunks))):
        print(f"Chunk {i} | Section: '{final_chunks[i].metadata['section'][0:20]}...'")
        # print(f"Preview: {final_chunks[i].page_content[:50]}...\n")
    print("----------------------\n")

    ingest_to_pinecone(final_chunks)

    if traffic_source != "Admin":
        send_upload_notification(character_name, clean_url, session_id)
    else:
        print("🔔 skipped upload notification")

    return f"✅ Successfully ingested: {clean_url} for '{character_name}'"
