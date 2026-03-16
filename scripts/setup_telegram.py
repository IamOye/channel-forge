"""
setup_telegram.py — Telegram notification setup helper for ChannelForge.

Run:
    python scripts/setup_telegram.py

If both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in .env, a test
message is sent to confirm the integration is working.  Otherwise, clear
step-by-step instructions are printed.
"""

import os
import sys
from pathlib import Path

# Allow running from project root or scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

_INSTRUCTIONS = """
To set up Telegram notifications:

1. Open Telegram and search for @BotFather
2. Send /newbot and follow instructions
3. Copy the bot token provided
4. Add to .env: TELEGRAM_BOT_TOKEN=your_token

5. Start a chat with your new bot
6. Send any message to it
7. Visit this URL to get your chat ID:
   https://api.telegram.org/bot{TOKEN}/getUpdates
8. Copy the chat_id from the response
9. Add to .env: TELEGRAM_CHAT_ID=your_chat_id

10. Run this script again to send a test message
"""


def main() -> None:
    token   = os.getenv("TELEGRAM_BOT_TOKEN",  "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print(_INSTRUCTIONS)
        if not token:
            print("  ❌  TELEGRAM_BOT_TOKEN is not set in .env")
        if not chat_id:
            print("  ❌  TELEGRAM_CHAT_ID is not set in .env")
        sys.exit(1)

    from src.notifications.telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier(token=token, chat_id=chat_id)
    ok = notifier.send(
        "✅ Telegram notifications configured successfully for ChannelForge!"
    )
    if ok:
        print("✅ Test message sent! Check your Telegram chat.")
    else:
        print("❌ Failed to send test message. Check your token and chat_id.")
        sys.exit(1)


if __name__ == "__main__":
    main()
