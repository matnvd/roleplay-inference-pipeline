import os

import requests
from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


# for notifying pinecone db updates to avoid overusage
def send_upload_notification(character_name, source_url, session_id):
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ No Discord Webhook URL found in .env, skipping notification.")
        return

    payload = {
        "username": "Project RIP Notifier Bot",
        "embeds": [
            {
                "title": "🚀 New Character Uploaded!",
                "color": 5763719,  # green
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
