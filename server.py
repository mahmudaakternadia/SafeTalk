import asyncio
import websockets
import json
import os
from datetime import datetime

import requests as http_requests
from openai import OpenAI
from better_profanity import profanity

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from dotenv import load_dotenv
load_dotenv()

# === Config ===
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
HF_API_KEY = os.getenv("HF_API_KEY")
client = OpenAI()

HOST = "127.0.0.1"
PORT = 12345
FLAGGED_LOG_PATH = "flagged_messages.log"

# === State ===
clients = []
users = {}   # ws -> {email,name,pic}
unsafe_counts = {}
rate_limit_counts = {}
last_message_times = {}
banned_emails = set()

# === Profanity ===
profanity.load_censor_words()


def verify_google_token(token: str):
    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10  
        )
        return {
            "email": idinfo.get("email"),
            "name": idinfo.get("name", idinfo.get("email", "User")),
            "pic": idinfo.get("picture", "")
        }
    except Exception as e:
        print("Token verification failed:", e)
        return None


def log_flagged(email, message, reason):
    try:
        with open(FLAGGED_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} | {email}: {message} | {reason}\n")
    except:
        pass


def check_profanity_text(text):
    return profanity.contains_profanity(text)


def check_hf_toxicity(text):
    if not HF_API_KEY:
        return True, "No moderation API key."
    try:
        r = http_requests.post(
            "https://api-inference.huggingface.co/models/unitary/toxic-bert",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": text},
            timeout=10,
        )
        result = r.json()
        toxic_score = 0.0
        if isinstance(result, list):
            block = result[0] if result and isinstance(result[0], list) else result
            for item in block:
                if item.get("label") == "toxic":
                    toxic_score = item.get("score", 0.0)
        if toxic_score > 0.5:
            return False, f"Toxicity flagged | Score: ({toxic_score:.2f}) | Blocked by Hugging Face Toxicity"
        return True, "OK"
    except Exception as e:
        print("HF moderation error:", e)
        return True, "HF unavailable"


def check_safe(text):
    if check_profanity_text(text):
        return False, "Profanity detected"
    return check_hf_toxicity(text)


async def broadcast_json(obj, except_ws=None):
    msg = json.dumps(obj)
    for c in clients[:]:
        if c != except_ws and c.open:
            try:
                await c.send(msg)
            except websockets.exceptions.ConnectionClosed:
                await remove_client(c)


async def broadcast_user_list():
    online = [{"name": u["name"], "email": u["email"], "pic": u["pic"]} for u in users.values()]
    await broadcast_json({"type": "user_list", "users": online})


async def broadcast_system(content):
    await broadcast_json({"type": "system", "content": content})


async def remove_client(ws):
    if ws in clients:
        u = users.pop(ws, None)
        clients.remove(ws)
        unsafe_counts.pop(ws, None)
        rate_limit_counts.pop(ws, None)
        last_message_times.pop(ws, None)
        if u:
            await broadcast_system(f"{u['name']} left the chat.")
        await broadcast_user_list()


async def handle_client(ws):
    clients.append(ws)
    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except:
                continue

            # --- First message must be auth ---
            if data.get("type") == "auth":
                token = data.get("token")
                user = verify_google_token(token)
                if not user or not user.get("email"):
                    await ws.send(json.dumps({"type": "kick", "content": "Invalid Google Sign-In"}))
                    await ws.close()
                    return

                email = user["email"]

                # banned?
                if email in banned_emails:
                    await ws.send(json.dumps({"type": "kick", "content": "You are banned."}))
                    await ws.close()
                    return

                # Already logged in?
                if email in [u["email"] for u in users.values()]:
                    await ws.send(json.dumps({"type": "kick", "content": "⚠️ This account is already active."}))
                    await ws.close()
                    return

                users[ws] = user
                unsafe_counts[ws] = 0
                rate_limit_counts[ws] = 0
                await broadcast_system(f"{user['name']} joined the chat!")
                await broadcast_user_list()
                await ws.send(json.dumps({"type": "system", "content": "✅ Connected to SafeTalk"}))
                continue

            # Must be authed
            if ws not in users:
                await ws.send(json.dumps({"type": "kick", "content": "Not authenticated."}))
                await ws.close()
                return

            user = users[ws]
            email = user["email"]
            name = user["name"]
            pic = user["pic"]
            now = datetime.utcnow()

            # --- Chat messages ---
            if data.get("type") == "chat":
                msg = data.get("content", "").strip()
                if not msg:
                    continue
                if len(msg) > 200:
                    await ws.send(json.dumps({"type": "warning", "content": "Message too long."}))
                    continue

                # Rate limiting
                last = last_message_times.get(ws)
                if last and (now - last).total_seconds() < 1:
                    rate_limit_counts[ws] += 1
                    await ws.send(json.dumps({"type": "warning", "content": "Too fast! Slow down."}))
                    if rate_limit_counts[ws] >= 5:
                        banned_emails.add(email)
                        await ws.send(json.dumps({"type": "kick", "content": "Banned for spamming."}))
                        await ws.close()
                        return
                    continue
                last_message_times[ws] = now

                # AI command
                if msg.lower().startswith("/ai "):
                    prompt = msg[4:].strip()
                    safe, reason = check_safe(prompt)
                    if not safe:
                        unsafe_counts[ws] += 1
                        log_flagged(email, prompt, reason)
                        await ws.send(json.dumps({"type": "warning", "content": f"⚠️ AI prompt blocked: {reason}"}))
                        if unsafe_counts[ws] >= 3:
                            banned_emails.add(email)
                            await ws.send(json.dumps({"type": "kick", "content": "Banned for unsafe inputs."}))
                            await ws.close()
                        continue
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful, safe chatbot in SafeTalk."},
                                {"role": "user", "content": prompt},
                            ],
                            max_tokens=150,
                        )
                        ai_reply = response.choices[0].message.content.strip()
                    except Exception as e:
                        ai_reply = f"Error: {e}"
                    await broadcast_json({
                        "type": "chat",
                        "content": f"🤖 {ai_reply}",
                        "sender": "AI",
                        "name": "AI",
                        "pic": "",
                        "timestamp": now.isoformat(),
                    })
                    continue

                # Normal user message
                safe, reason = check_safe(msg)
                if safe:
                    await broadcast_json({
                        "type": "chat",
                        "content": msg,
                        "sender": email,
                        "name": name,
                        "pic": pic,
                        "timestamp": now.isoformat(),
                    })
                else:
                    unsafe_counts[ws] += 1
                    log_flagged(email, msg, reason)
                    await ws.send(json.dumps({"type": "warning", "content": f"⚠️ Blocked: {reason}"}))
                    if unsafe_counts[ws] >= 3:
                        banned_emails.add(email)
                        await ws.send(json.dumps({"type": "kick", "content": "Banned for repeated unsafe messages."}))
                        await ws.close()
                        return

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await remove_client(ws)


async def main():
    print(f"🚀 SafeTalk running on ws://{HOST}:{PORT}")
    async with websockets.serve(handle_client, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutting down.")


