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

# The specific Wiki page we're ripping data from... implement multiple sources in future
# TARGET_URL = "https://leagueoflegends.fandom.com/wiki/Jinx/Arcane"
# TARGET_URL = "https://arcane.fandom.com/wiki/Jinx"
# TARGET_URL = "https://how-i-met-your-mother.fandom.com/wiki/Barney_Stinson"
INDEX_NAME = "project-rip"


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
        response.raise_for_status()  # Crashes nicely if link is dead (404)
    except Exception as e:
        print(f"❌ Error fetching page: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # FINDING CONTENT
    # On Fandom Wikis, the actual article is always inside <div class="mw-parser-output">.
    # ignore sidebars, ads, and footers.
    content_div = soup.find("div", {"class": "mw-parser-output"})

    if not content_div:
        print(
            "❌ Could not find main content div. The Wiki structure might have changed."
        )
        return None

    # DATA CLEANING
    # We only want paragraphs <p>, lists <ul>, and headers <h>
    clean_text = ""
    section_map = []  # storing: {'start':0, 'end':500, 'header': 'Intro'}

    current_header = "Introduction"
    current_idx = 0

    tags_to_scrape = ["p", "h2", "h3", "h4", "ul"]  # add more later if needed

    for element in content_div.find_all(tags_to_scrape, recursive=False):
        tag = element.name
        text = element.get_text().strip()

        if not text:
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
    return clean_text, section_map


# CHUNKING
# nlp pre-processing: splits long txt into chunks for vector db
def chunk_text(raw_text, section_map, source_url, character_name):
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

        # find dominant section for each chnuk
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
def check_if_url_exists(url):
    print(f"🔍 Checking if {url} is already in the database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    dummy_vector = [0.0] * 1536

    try:
        results = vectorstore.similarity_search_by_vector_with_score(
            embedding=dummy_vector, k=1, filter={"source": url}
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
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    print(f"🚀 Uploading {len(chunks)} chunks to Pinecone index '{INDEX_NAME}'...")

    try:
        # # uploads to pinecone (sends chunks to openai to get vectors and forwards to pinecone)
        vectorstore.add_documents(documents=chunks)
        print("✅ Success! Data is now live in the Vector Database.")
    except Exception as e:
        print(f"❌ Upload failed: {e}")


# MAIN (exported) FUNCTION
def ingest_fandom_wiki(url, character_name):
    if not url:
        return "⚠️ No URL provided."

    if not character_name:
        character_name = "Unknown Character"  # add counter for num of characters

    parsed = urlparse(url)
    clean_url = urlunparse(parsed._replace(fragment="")).rstrip("/")

    # Debug: Show user what happened
    if url != clean_url:
        print(f"🧹 Normalized URL: '{url}' -> '{clean_url}'")

    if check_if_url_exists(clean_url):
        return f"⚠️ Skipping ingestion. '{clean_url}' is already in the DB."

    print(f"🔗 New URL detected for character '{character_name}': {clean_url}")

    # run extractor
    raw_data, map_data = rip_wiki_content(clean_url)

    if not raw_data:
        return "❌ Failed to scrape content."

    # run transformer
    final_chunks = chunk_text(raw_data, map_data, clean_url, character_name)
    if not final_chunks:
        return "❌ Failed to create chunks"

    print("\n--- METADATA (preview): ---")
    for i in range(min(5, len(final_chunks))):
        print(f"Chunk {i} | Section: '{final_chunks[i].metadata['section'][0:20]}...'")
        # print(f"Preview: {final_chunks[i].page_content[:50]}...\n")
    print("----------------------\n")

    ingest_to_pinecone(final_chunks)

    return f"✅ Successfully ingested: {clean_url} for '{character_name}'"
