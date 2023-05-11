# app.py
from fastapi import FastAPI
from chatbot import initialize_chatbot
from pydantic import BaseModel
import sqlite3
from config import config

class Question(BaseModel):
    question: str

class Feedback(BaseModel):
    question: str
    answer: str
    feedback: int

app = FastAPI()

chatbot = initialize_chatbot(config)

@app.on_event("startup")
async def startup_event():
    global chatbot
    chatbot = initialize_chatbot(config)

@app.post("/ask/")
async def ask_bot(question: Question):
    response = chatbot({"question": question.question, "chat_history": []})
    return {"answer": response["answer"]}

@app.post("/feedback")
async def store_feedback(feedback: Feedback):
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute("INSERT INTO feedback (question, answer, feedback) VALUES (?, ?, ?)",
              (feedback.question, feedback.answer, feedback.feedback))
    conn.commit()
    conn.close()
    return {"message": "Feedback stored successfully"}

