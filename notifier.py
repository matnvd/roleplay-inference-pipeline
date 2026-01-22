import os

import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


# for notifying pinecone db updates to avoid overusage
def send_upload_notification(character_name, source_url, session_id):
    if not SLACK_WEBHOOK_URL:
        print("⚠️ No Slack Webhook URL found, skipping notification.")
        return

    payload = {
        "text": f"🚀 New Character: {character_name}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🚀 New Character Uploaded!"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*🎭 Character:*\n{character_name}"},
                    {
                        "type": "mrkdwn",
                        "text": f"*🔗 Source:*\n<{source_url}|Wiki Link>",
                    },
                ],
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Session ID:* `{session_id}` | _Project RIP Vector Database_",
                    }
                ],
            },
        ],
    }

    try:
        response = requests.post(SLACK_WEBHOOK_URL.strip(), json=payload, timeout=10)
        response.raise_for_status()
        print("🔔 Character upload Slack notification sent!")
    except Exception as e:
        print(f"❌ Failed to send Slack upload notification: {e}")


# for notifying ab diff traffic sources
def send_traffic_notification(traffic_source, session_id):
    if not SLACK_WEBHOOK_URL:
        print("⚠️ No Slack Webhook URL found, skipping notification.")
        return

    payload = {
        "text": f"🔓 Login Alert: {traffic_source}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🔓 New Login Alert!"},
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*🌐 Traffic Source:*\n{traffic_source}",
                    },
                    {"type": "mrkdwn", "text": f"*🆔 Session ID:*\n`{session_id}`"},
                ],
            },
            {"type": "divider"},
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": "Project RIP Security Monitor"}
                ],
            },
        ],
    }

    try:
        response = requests.post(SLACK_WEBHOOK_URL.strip(), json=payload, timeout=10)
        response.raise_for_status()
        print("🔔 Traffic source Slack notification sent!")
    except Exception as e:
        print(f"❌ Failed to send Slack traffic notification: {e}")
