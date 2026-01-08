import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# The specific Wiki page we're ripping data from... implement multiple sources in future
# TARGET_URL = "https://leagueoflegends.fandom.com/wiki/Jinx/Arcane"
TARGET_URL = "https://arcane.fandom.com/wiki/Jinx"
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
    tags_to_scrape = ["p", "h2", "h3", "h4", "ul"]  # add more later if needed

    for element in content_div.find_all(tags_to_scrape, recursive=False):
        text = element.get_text().strip()
        if text:
            clean_text += text + "\n\n"

    print(f"✅ Successfully extracted {len(clean_text)} characters of raw text.")
    return clean_text


# CHUNKING
# nlp pre-processing: splits long txt into chunks for vector db
def chunk_text(raw_text):
    print("🔪 Chunking text...")

    # splits Paragraph by (\n\n) first, then Sentence (.), then Word.
    # This prevents cutting a sentence in half, which confuses the AI.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # ~200-300 words per chunk, longer better for narrative context?
        chunk_overlap=400,  # Overlap ensures context flows across chunks
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = text_splitter.create_documents([raw_text])

    print(f"✅ Split text into {len(chunks)} vector-ready chunks.")
    return chunks


# embed and send chunks to pinecone
def ingest_to_pinecone(chunks):
    print(f"🚀 Uploading {len(chunks)} chunks to Pinecone index '{INDEX_NAME}'...")

    # once working look at mteb leaderboard and change model? llama?
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Nukes db everytime to avoid duplicates
    print("🗑️ Clearing previous vectors...")
    try:
        # This deletes everything in the default namespace
        vectorstore.delete(delete_all=True)
        print("   ✅ Index cleared.")
    except Exception as e:
        print(f"   ⚠️ Could not clear index (might be empty): {e}")

    print(f"🚀 Uploading {len(chunks)} new chunks...")

    try:
        # # uploads to pinecone (sends chunks to openai to get vectors and forwards to pinecone)
        vectorstore.add_documents(documents=chunks)
        print("✅ Success! Data is now live in the Vector Database.")
    except Exception as e:
        print(f"❌ Upload failed: {e}")


# MAIN FUNCTION
if __name__ == "__main__":
    # 1. Run the Extractor
    raw_data = rip_wiki_content(TARGET_URL)

    if raw_data:
        # 2. Run the Transformer
        final_chunks = chunk_text(raw_data)

        # creating unique ID's
        # uuids = [str(uuid4()) for _ in range(len(final_chunks))]

        # print("\n chunk 0:")
        # print(final_chunks[0].page_content)
        # print("----------------------------------------------")

        ingest_to_pinecone(final_chunks)
