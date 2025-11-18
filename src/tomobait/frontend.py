from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from tomobait.app import run_agent_chat
from typing import Annotated

app = FastAPI()

chat_history = []

@app.get("/")
async def get():
    chat_html = "".join([f"<p><b>{sender}:</b> {msg}</p>" for sender, msg in chat_history])
    return HTMLResponse(content=f"""
        <html>
            <head>
                <title>TomoBait</title>
            </head>
            <body>
                <h1>TomoBait</h1>
                <div id="chat">
                    {chat_html}
                </div>
                <form action="/chat" method="post">
                    <input type="text" name="message" autofocus>
                    <button type="submit">Send</button>
                </form>
            </body>
        </html>
    """, status_code=200)

@app.post("/chat")
async def chat(message: Annotated[str, Form()]):
    chat_history.append(("User", message))
    agent_response = run_agent_chat(message)
    chat_history.append(("Agent", agent_response))
    
    chat_html = "".join([f"<p><b>{sender}:</b> {msg}</p>" for sender, msg in chat_history])
    return HTMLResponse(content=f"""
        <html>
            <head>
                <title>TomoBait</title>
            </head>
            <body>
                <h1>TomoBait</h1>
                <div id="chat">
                    {chat_html}
                </div>
                <form action="/chat" method="post">
                    <input type="text" name="message" autofocus>
                    <button type="submit">Send</button>
                </form>
            </body>
        </html>
    """, status_code=200)
