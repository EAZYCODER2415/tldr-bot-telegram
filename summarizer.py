'''
summarizer.py - Python library where all summarizing tasks are processed through
the Qwen LLM API

Instead of the Telegram bot typing its messages, it only has to run the functions here
and have completed AI summary responses ready for submission. The following commands in this
library has to handle the processing of all message buffer content and prepare AI-generated responses.

'''

import os, time
from openai import OpenAI, APIError

import asyncio
import httpx
import os
import logging

logger = logging.getLogger(__name__)

# DashScope provides an OpenAI-compatible endpoint
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/EAZYCODER2415/tg-niche-summarizer_bot",  # Required by OpenRouter
        "X-Title": "SMU Niche Summarizer Bot" # Required by OpenRouter
    }
)

# Load system prompt from txt file
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "system_instruction.txt")
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    system_instruction = f.read()

free_models = [
    "openrouter/free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-31b-it:free"
]

plugins = [{"id": "web"}]

async def openrouter_heartbeat_loop(context=None):
    """
    Pings OpenRouter every X seconds (default: 5 minutes) with a tiny 
    completion request to keep the TCP connection alive and warm.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("⚠️ OPENROUTER_API_KEY not found. Heartbeat disabled.")
        return

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/EAZYCODER2415/tldr-bot-telegram",
        "X-Title": "TLDRFlow Keepalive"
    }
    # Minimal payload to consume almost zero tokens/cost
    payload = {
        "model": "qwen/qwen-2.5-72b-instruct",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1
    }

    # Give the bot time to fully boot up before the first ping
    await asyncio.sleep(10)
    
    interval_seconds = 300  # 5 minutes
    try:
        # Use a fresh short-lived client for the heartbeat ping
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                logger.info("🟢 OpenRouter heartbeat successful (connection kept warm).")
            else:
                logger.warning(f"⚠️ OpenRouter heartbeat returned status: {response.status_code}")
    except Exception as e:
        # Catch timeouts/network drops silently so it never crashes the bot
        await logger.debug(f"ℹ️ Heartbeat ping skipped due to temporary network/idle state: {e}")

def summarizeLLMtool(prompt: str, file_url: str = None, link: str = None, language: str = None) -> str:
    """
    Summarizes chat messages or processes multimodal input using Qwen on OpenRouter.
    
    :param prompt: The formatted text messages/chat transcript to summarize.
    :param image_url: (Optional) The formatted list of captions w/ image attachments.
    :param model: The model name to use ("qwen-max" for text, "qwen-vl-max" for images).
    :return: Generated summary string.
    """

    # Base text model (for first try)
    model = "qwen/qwen-2.5-72b-instruct"

    # Separate URLs from raw document filenames
    urls = []
    doc_names = []

    if file_url:
        # Split by whitespace/newlines if file_url contains multiple bundled items
        items = file_url.split()
        for item in items:
            if item.startswith("http://") or item.startswith("https://"):
                urls.append(item)
            else:
                doc_names.append(item)

    # Base text message with any hidden document filenames included as context
    combined_text = f"Message: {prompt}"
    
    if language not in ["auto"]:
        combined_text += f"\nSummary Language: {language}, "
    
    if doc_names:
        combined_text += f"\nAttachments: {', '.join(doc_names)}"

    # Add first message link for processing
    if link:
        combined_text += f"\nFirst message link: {link}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Build content list dynamically
            content = [
                {"type": "text", "text": combined_text}
            ]

            # Append image URLs cleanly
            if urls:
                model = "google/gemma-4-26b-a4b-it"  # Switch to vision model
                for url in urls:
                    content.append({"type": "image_url", "image_url": {"url": urls}})
            if urls:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.2,  # Forces adherence to prompt formatting
                    top_p=0.9,
                    messages=[
                        {
                            "role": "system",
                            "content": system_instruction,
                        },
                        {"role": "user", "content": content}
                    ],
                    extra_body={
                        "models": free_models,
                        "plugins": plugins
                    },
                    timeout=60.0
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.2,  # Forces adherence to prompt formatting
                    top_p=0.9,
                    messages=[
                        {
                            "role": "system",
                            "content": system_instruction,
                        },
                        {"role": "user", "content": content}
                    ],
                    extra_body={
                        "models": free_models,
                        "provider": {
                            "order": ["DeepInfra", "Together"],
                            "allow_fallbacks": False
                        },
                        "plugins": plugins
                    },
                    timeout=60.0
                )
            return response.choices[0].message.content

        except APIError as e:
            print(f"[Attempt {attempt + 1}/{max_retries}] OpenRouter Error: {e}")
            if attempt == max_retries - 1:
                return f"⚠️ OpenRouter Error: Request failed after {max_retries} attempts."
            
            time.sleep(2 ** attempt + 1)

        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"⚠️ Summary unavailable: {str(e)}"

def checkForTopic(message: str, topic: str, file_url: str = None) -> bool:
    """
    Checks message for a specific topic input. Will return a boolean.
    """

    # Base text model (for first try)
    model = "qwen/qwen-2.5-72b-instruct"

    # Separate URLs from raw document filenames
    urls = []
    doc_names = []

    if file_url:
        # Split by whitespace/newlines if file_url contains multiple bundled items
        items = file_url.split()
        for item in items:
            if item.startswith("http://") or item.startswith("https://"):
                urls.append(item)
            else:
                doc_names.append(item)

    # Base text message with any hidden document filenames included as context
    combined_text = f"Message: {message}"
    if doc_names:
        combined_text += f"\nAttachments: {', '.join(doc_names)}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Build content list dynamically
            content = [
                {"type": "topic", "topic": topic},
                {"type": "text", "text": combined_text}
            ]

            # Append image URLs cleanly
            if urls:
                model = "google/gemma-4-26b-a4b-it"  # Switch to vision model
                for url in urls:
                    content.append({"type": "image_url", "image_url": {"url": urls}})

            if urls:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.2,  # Forces adherence to prompt formatting
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are detecting whether the given message (or attached files) is related to the topic: {topic}. Output ONLY 'True' or 'False'."
                        },
                        {"role": "user", "content": content}
                    ],
                    extra_body={
                        "models": free_models,
                    },
                    timeout=40.0
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.2,  # Forces adherence to prompt formatting
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are detecting whether the given message (or attached files) is related to the topic: {topic}. Output ONLY 'True' or 'False'."
                        },
                        {"role": "user", "content": content}
                    ],
                    extra_body={
                        "models": free_models,
                        "provider": {
                            "order": ["DeepInfra", "Together"],
                            "allow_fallbacks": False
                        }
                    },
                    timeout=40.0
                )

            return response.choices[0].message.content.strip() == "True"

        except APIError as e:
            print(f"[Attempt {attempt + 1}/{max_retries}] OpenRouter Error: {e}")
            if attempt == max_retries - 1:
                return f"⚠️ OpenRouter Error: Request failed after {max_retries} attempts."
            
            time.sleep(2 ** attempt + 1)

        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"⚠️ checkForTopic unavailable: {str(e)}"

# google/gemma-4-26b-a4b-it:free
def validate_language(input: str=None) -> str:
    """
    Lightweight language validator using a fast OpenRouter model.
    """
    model = "google/gemma-4-26b-a4b-it:free" # Or your preferred fast free model
    
    system_prompt = (
        "You are an input validator for a language configuration setting.\n"
        "Analyze the user's text and determine if it represents a real language, dialect, "
        "or recognizable language blend (e.g., Singlish, Hinglish, Taglish).\n"
        "Return the original user input if it is immediately recognizable as a validated language, if not and\n"
        "the input implies an actual language, try to reword and standardize it in English\n"
        "If all else fails, return only 'False'.\n"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Build content list dynamically
            content = [
                {"type": "text", "text": input},
            ]

            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                top_p=0.9,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {"role": "user", "content": content}
                ],
                extra_body={
                    "models": free_models,
                    "provider": {
                        "order": ["DeepInfra", "Together"],
                        "allow_fallbacks": False
                    },
                    "plugins": plugins
                },
                timeout=30.0
            )
            return response.choices[0].message.content

        except APIError as e:
            print(f"[Attempt {attempt + 1}/{max_retries}] OpenRouter Error: {e}")
            if attempt == max_retries - 1:
                return f"⚠️ OpenRouter Error: Request failed after {max_retries} attempts."
            
            time.sleep(2 ** attempt + 1)

        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"⚠️ Could not check language: {str(e)}"