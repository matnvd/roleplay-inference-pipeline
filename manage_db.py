import os
import time
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Configuration
INDEX_NAME = "project-rip"
API_KEY = os.getenv("PINECONE_API_KEY")


def main():
    if not API_KEY:
        print("❌ Error: PINECONE_API_KEY not found in .env file.")
        return

    # 1. Initialize Connection
    try:
        pc = Pinecone(api_key=API_KEY)
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()

        print("\n==========================================")
        print(f" 🗄️  DATABASE MANAGER: {INDEX_NAME}")
        print("==========================================")
        print(f"📊 Current Total Vectors: {stats.total_vector_count}")
        print("==========================================\n")

    except Exception as e:
        print(f"❌ Could not connect to Pinecone: {e}")
        return

    # 2. Menu Options
    print("What would you like to do?")
    print("1. 🗑️ Delete a specific Source URL")
    print("2. ☢️ NUKE IT (Delete ALL data)")
    print("3. 👦 Delete a specific character")
    print("x. ❌ Exit")

    choice = input("\nEnter choice (1, 2, 3, or x): ").strip()

    # --- OPTION 1: DELETE SPECIFIC URL ---
    if choice == "1":
        target_url = input("\nEnter the full Source URL to remove: ").strip()

        parsed = urlparse(target_url)
        clean_url = urlunparse(parsed._replace(fragment="")).rstrip("/")

        # Debug: Show user what happened
        if target_url != clean_url:
            print(f"🧹 Normalized URL: '{target_url}' -> '{clean_url}'")

        if not clean_url:
            print("⚠️ No URL entered. Aborting.")
            return

        print(f"\n🔍 Deleting vectors where metadata['source'] == '{clean_url}'...")
        try:
            # The Magic Line: Deletes only vectors matching the filter
            index.delete(filter={"source": clean_url})
            print(f"✅ Success! All chunks from '{clean_url}' have been removed.")

            # Verify update
            time.sleep(2)  # Give Pinecone a moment to update stats
            new_stats = index.describe_index_stats()
            print(f"📊 New Total Vector Count: {new_stats.total_vector_count}")

        except Exception as e:
            print(f"❌ Error deleting data: {e}")

    # --- OPTION 2: DELETE ALL ---
    elif choice == "2":
        print(f"Deleting ALL {stats.total_vector_count} vectors.")

        confirm = input("Type 'DELETE' exactly to confirm: ").strip()

        if confirm == "DELETE":
            try:
                index.delete(delete_all=True)
                print("\n✅ Index cleared. The database is empty.")
            except Exception as e:
                print(f"❌ Error clearing index: {e}")
        else:
            print("🚫 Confirmation failed. Operation cancelled.")

    # filter by character
    elif choice == "3":
        target_char = input("\nEnter the full character name to remove: ").strip()

        print(
            f"\n🔍 Deleting vectors where metadata['character'] == '{target_char}'..."
        )
        try:
            # The Magic Line: Deletes only vectors matching the filter
            index.delete(filter={"character": target_char})
            print(f"✅ Success! All chunks from '{target_char}' have been removed.")

            # Verify update
            time.sleep(2)  # Give Pinecone a moment to update stats
            new_stats = index.describe_index_stats()
            print(f"📊 New Total Vector Count: {new_stats.total_vector_count}")

        except Exception as e:
            print(f"❌ Error deleting data: {e}")

    # --- OPTION 3: EXIT ---
    else:
        print("\n👋 Exiting.")


if __name__ == "__main__":
    main()
