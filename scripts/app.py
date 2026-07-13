# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from qa import get_answers  # the new QA module

app = FastAPI(title="Milvus QA API")

# Request model for single question
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5  # number of chunks to retrieve

# Request model for batch questions
class BatchRequest(BaseModel):
    questions: List[str]
    top_k: Optional[int] = 5

# Health check
@app.get("/")
def read_root():
    return {"message": "Milvus QA API is running!"}

# Single question endpoint
@app.post("/search")
def search_question(req: QuestionRequest):
    results = get_answers([req.question], top_k=req.top_k)
    return results[0]  # only one question

# Batch questions endpoint
@app.post("/batch_search")
def batch_search(req: BatchRequest):
    results = get_answers(req.questions, top_k=req.top_k)
    return {"results": results}
