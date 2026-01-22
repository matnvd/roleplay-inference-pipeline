import re
from urllib.parse import unquote, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from notifier import send_upload_notification

load_dotenv()

INDEX_NAME = "project-rip"


# CONTENT COLLECTION
# web scrapes and cleans data to inject into db
# Fetches the raw HTML and isolates the main article text.
def rip_wiki_content(url):
    print(f"⚡ connecting to {url}...")

    html_content = ""

    # wikipedia: using official api
    if "wikipedia.org" in url:
        try:
            # 1. Parse the URL to get Language and Page Title
            parsed = urlparse(url)
            # e.g., en.wikipedia.org -> "en"
            lang = parsed.netloc.split(".")[0]
            # handles path parsing
            page_title = unquote(parsed.path.split("/")[-1])

            print(f"    📖 Detected Wikipedia API. Fetching '{page_title}' ({lang})...")

            api_url = f"https://{lang}.wikipedia.org/w/api.php"
            params = {
                "action": "parse",
                "page": page_title,
                "format": "json",
                "prop": "text",
                "redirects": 1,
                "disabletoc": 1,
            }

            # PASS HEADERS HERE TO FIX 403 ERROR
            headers = {
                "User-Agent": "ProjectRip/1.0 (your_email@example.com) python-requests/2.31"
            }
            response = requests.get(api_url, params=params, headers=headers)

            if response.status_code == 403:
                print("❌ Wikipedia blocked the request. Check your User-Agent header.")
                return None, None

            response.raise_for_status()
            data = response.json()

            if "error" in data:
                print(f"❌ API Error: {data['error'].get('info')}")
                return None, None

            html_content = data["parse"]["text"]["*"]

        except Exception as e:
            print(f"❌ Error fetching via Wikipedia API: {e}")
            return None, None
    else:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            response = requests.get(url, headers=headers)
            # print(f"response.text: {response.text}")
            response.raise_for_status()  # Crashes nicely if link is dead (404)
            html_content = response.text
        except Exception as e:
            print(f"❌ Error fetching page: {e}")
            return None, None

    soup = BeautifulSoup(html_content, "html.parser")
    # print(f"soup.text: {soup.text}")
    infobox_text, image_url = extract_infobox(soup)

    if image_url:
        print(f"🖼️  Successfully scraped image: {image_url}")
    else:
        print("⚠️ No profile image found in infobox.")

    # print(f"soup.title: {soup.title.string if soup.title else 'No Title'}")

    # FINDING CONTENT
    # On Fandom Wikis, article usually inside <div class="mw-parser-output">.
    # ignore sidebars, ads, and footers.
    content_div = soup.find("div", {"class": "mw-parser-output"})

    # text extraction
    if not content_div and "wikipedia.org" in url:
        content_div = soup

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
    # print(f"ℹ️infobox_text: {infobox_text}")
    clean_text = infobox_text
    section_map = []  # storing: {'start':0, 'end' :500, 'header': 'Intro'}

    # for source metadata
    if infobox_text:
        section_map.append({"start": 0, "end": len(infobox_text), "header": "Infobox"})

    current_header = "Introduction"
    current_idx = len(clean_text)

    tags_to_scrape = [
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "ul",
    ]  # add more later if needed

    # IGNORE LIST: Skip these sections entirely to reduce noise
    ignored_headers = [
        "See also",
        "References",
        "External links",
        "Notes",
        "Further reading",
        "Sources",
    ]
    skip_current_section = False

    for element in content_div.find_all(tags_to_scrape):
        tag = element.name
        text = element.get_text(" ", strip=True)

        # --- HEADER CLEANING ---
        if tag in ["h2", "h3", "h4"]:
            # 1. Remove [edit], [1], and empty []
            # This regex matches anything inside brackets and removes it
            text = re.sub(r"\[.*?\]", "", text).strip()

            # 2. Check if we should skip this section
            if any(ignored in text for ignored in ignored_headers):
                skip_current_section = True
                continue  # Skip adding this header
            else:
                skip_current_section = False

        # If we are in an ignored section (like References), skip this paragraph
        if skip_current_section:
            continue

        # --- TEXT BODY CLEANING ---
        # Remove citation numbers like [1], [25] from paragraphs
        text = re.sub(r"\[\d+\]", "", text)

        if not text or len(text) < 3:
            continue

        # add header info in front of text
        if tag in ["h2", "h3", "h4"]:  # add h4?
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

    return clean_text, section_map, image_url


# extracting infobox basics for wiki links
def extract_infobox(soup):
    infobox_text = ""

    # 1. Find the Infobox (Wikipedia uses .infobox, Fandom uses .portable-infobox)
    infobox = soup.find("table", class_=lambda x: x and "infobox" in x)
    if not infobox:
        infobox = soup.find("aside", {"class": "portable-infobox"})

    if infobox:
        # --- IMAGE EXTRACTION ---

        img_tag = infobox.find("img", {"class": "mw-file-element"})

        if not img_tag:
            img_tag = infobox.find("img", {"class": "pi-image-thumbnail"})

        if not img_tag:
            wrapper = infobox.find(class_=["infobox-image", "view-image"])
            if wrapper:
                img_tag = wrapper.find("img")

        if not img_tag:
            img_tag = infobox.find("img")

        # Extract & Clean URL
        if img_tag and img_tag.get("src"):
            raw_src = img_tag["src"]

            if raw_src.startswith("//"):
                raw_src = "https:" + raw_src

            if "/revision/" in raw_src:
                raw_src = raw_src.split("/revision/")[0]

            image_url = raw_src

        # TEXT EXTRACTION
        infobox_text += "== QUICK FACTS (Infobox) ==\n"

        # WIKIPEDIA STYLE (Rows with th/td)
        rows = infobox.find_all("tr")
        for row in rows:
            th = row.find("th")  # Header (Key)
            td = row.find("td")  # Data (Value)

            if th and td:
                key = th.get_text(" ", strip=True)
                val = td.get_text(" ", strip=True)

                val = re.sub(r"\[\d+\]", "", val)
                val = val.replace("[]", "").strip()

                infobox_text += f"{key}: {val}\n"

            # Handle section headers inside infobox (e.g. "Personal details")
            elif th and not td:
                header_text = th.get_text(" ", strip=True)
                if len(header_text) > 2:
                    infobox_text += f"\n--- {header_text} ---\n"

        # FANDOM STYLE (divs)
        if not rows:  # If no table rows found, try Fandom structure
            for item in infobox.find_all("div", {"class": "pi-item"}):
                label = item.find("h3", {"class": "pi-data-label"})
                value = item.find("div", {"class": "pi-data-value"})
                if label and value:
                    infobox_text += (
                        f"{label.get_text(strip=True)}: {value.get_text(strip=True)}\n"
                    )

        infobox_text += "\n\n"

        # 2. Decompose (delete) the infobox from soup so the main loop doesn't scrape it again
        infobox.decompose()

    return infobox_text, image_url


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
        raw_data, map_data, image_url = rip_wiki_content(clean_url)
    except Exception:
        return "❌ Failed to scrape content."

    if not raw_data:
        print("❌ No data available at all")
        return "❌ Failed to scrape content"

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

    print(f"📸 img_url: {image_url}")
    return f"✅ Successfully ingested: {clean_url} for '{character_name}'", image_url
