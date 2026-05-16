import os, time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional
from contextlib import asynccontextmanager

from retriever import get_retriever
from agent import run_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the retriever index so /health passes cold-start check."""
    print("Warming up retriever...")
    r = get_retriever()
    print(f"Retriever ready: {len(r.catalog)} assessments indexed")
    yield


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL assessment selection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v):
        if v not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role: {v}. Must be 'user', 'assistant', or 'system'.")
        return v

class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages must not be empty")
        return v

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    messages = [m for m in messages if m["role"] != "system"]

    if len(messages) > 8:
        messages = messages[-8:]

    start = time.time()
    try:
        reply, recs_raw, end = run_agent(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    elapsed = time.time() - start
    if elapsed > 28:
        reply = reply or "I'm processing your request. Please try again momentarily."

    recs = [
        Recommendation(
            name=r.get("name", ""),
            url=r.get("url", ""),
            test_type=r.get("test_type", "A"),
        )
        for r in recs_raw
    ]

    return ChatResponse(
        reply=reply,
        recommendations=recs,
        end_of_conversation=end,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)