import os
import json
import uuid
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_engine import ProjectRAGEngine

load_dotenv()

app = FastAPI(title="Research Workspaces")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = "./storage/projects"
os.makedirs(STORAGE_DIR, exist_ok=True)

class CreateProjectReq(BaseModel):
    name: str
    context: str
    max_papers: int = 20
    year_from: Optional[int] = None

class SearchOpenAlexReq(BaseModel):
    query: str
    max_papers: int = 20
    year_from: Optional[int] = None

class EvidenceMapReq(BaseModel):
    topic: str

class ChatReq(BaseModel):
    message: str

class RenameChatReq(BaseModel):
    title: str

# --- Helper Functions ---
def get_project_meta_path(project_id: str):
    return os.path.join(STORAGE_DIR, project_id, "meta.json")

def load_project_meta(project_id: str):
    path = get_project_meta_path(project_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Project not found")
    with open(path, "r") as f:
        return json.load(f)

def save_project_meta(project_id: str, data: dict):
    path = get_project_meta_path(project_id)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def fetch_papers_task(project_id: str, context: str, max_papers: int, year_from: Optional[int]):
    try:
        engine = ProjectRAGEngine(project_id)
        count, query = engine.auto_fetch_papers(context, max_results=max_papers, year_from=year_from)

        meta = load_project_meta(project_id)
        meta["papers_count"] = meta.get("papers_count", 0) + count
        meta["last_query"] = query
        meta["status"] = "ready"
        meta.pop("fetch_error", None)
        save_project_meta(project_id, meta)
    except Exception as e:
        try:
            meta = load_project_meta(project_id)
            meta["status"] = "error"
            meta["fetch_error"] = str(e)
            save_project_meta(project_id, meta)
        except Exception:
            pass

# --- Endpoints ---

@app.get("/api/projects")
def list_projects():
    projects = []
    if os.path.exists(STORAGE_DIR):
        for pid in os.listdir(STORAGE_DIR):
            meta_path = get_project_meta_path(pid)
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    projects.append(json.load(f))
    return projects

@app.post("/api/projects")
def create_project(req: CreateProjectReq, background_tasks: BackgroundTasks):
    pid = str(uuid.uuid4())[:8]
    pdir = os.path.join(STORAGE_DIR, pid)
    os.makedirs(pdir, exist_ok=True)

    meta = {
        "id": pid,
        "name": req.name,
        "context": req.context,
        "papers_count": 0,
        "status": "fetching",
        "chats": [],
        "evidence_maps": {},     # New: persistence
        "study_materials": {}    # New: persistence
    }
    save_project_meta(pid, meta)
    background_tasks.add_task(fetch_papers_task, pid, req.context, req.max_papers, req.year_from)
    return meta

@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    return load_project_meta(project_id)

@app.post("/api/projects/{project_id}/upload-pdf")
async def upload_pdf(project_id: str, file: UploadFile = File(...)):
    meta = load_project_meta(project_id)
    engine = ProjectRAGEngine(project_id)
    
    contents = await file.read()
    chunks_added = engine.add_pdf(contents, file.filename)
    
    meta["papers_count"] += 1
    save_project_meta(project_id, meta)
    
    return {"message": f"Successfully ingested {file.filename}", "chunks": chunks_added}

@app.post("/api/projects/{project_id}/chats")
def create_chat(project_id: str, title: Optional[str] = "New Chat"):
    meta = load_project_meta(project_id)
    chat_id = str(uuid.uuid4())[:8]
    new_chat = {"id": chat_id, "title": title, "messages": []}
    meta["chats"].append(new_chat)
    save_project_meta(project_id, meta)
    return new_chat

# --- Rename Chat Endpoint ---
@app.patch("/api/projects/{project_id}/chats/{chat_id}")
def rename_chat(project_id: str, chat_id: str, req: RenameChatReq):
    meta = load_project_meta(project_id)
    chat = next((c for c in meta["chats"] if c["id"] == chat_id), None)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    chat["title"] = req.title.strip() or "Untitled Chat"
    save_project_meta(project_id, meta)
    return chat

# --- Streaming Message Endpoint ---
@app.post("/api/projects/{project_id}/chats/{chat_id}/message-stream")
def send_message_stream(project_id: str, chat_id: str, req: ChatReq):
    meta = load_project_meta(project_id)
    chat = next((c for c in meta["chats"] if c["id"] == chat_id), None)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Save user query
    chat["messages"].append({"role": "user", "content": req.message})
    save_project_meta(project_id, meta)

    engine = ProjectRAGEngine(project_id)

    def event_generator():
        full_response = ""
        try:
            for token in engine.chat_stream(req.message):
                full_response += token
                # Format as Server-Sent Events data payload
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Save completed AI response back to persistent storage
            meta_latest = load_project_meta(project_id)
            c_latest = next((c for c in meta_latest["chats"] if c["id"] == chat_id), None)
            if c_latest:
                c_latest["messages"].append({"role": "assistant", "content": full_response})
                save_project_meta(project_id, meta_latest)
                
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/projects/{project_id}/documents")
def list_project_documents(project_id: str):
    load_project_meta(project_id)
    try:
        engine = ProjectRAGEngine(project_id)
        return engine.list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load documents: {e}")

@app.delete("/api/projects/{project_id}/documents/{doc_name}")
def delete_project_document(project_id: str, doc_name: str):
    engine = ProjectRAGEngine(project_id)
    engine.delete_document(doc_name)
    
    meta = load_project_meta(project_id)
    docs = engine.list_documents()
    meta["papers_count"] = len(docs)
    save_project_meta(project_id, meta)
    
    return {"message": f"Deleted {doc_name}", "papers_count": len(docs)}


# Mode 1
@app.post("/api/projects/{project_id}/research/search")
def search_openalex(project_id: str, req: SearchOpenAlexReq):
    engine = ProjectRAGEngine(project_id)
    added_count = engine.search_and_add_openalex(req.query, max_results=req.max_papers, year_from=req.year_from)
    
    meta = load_project_meta(project_id)
    meta["papers_count"] = meta.get("papers_count", 0) + added_count
    save_project_meta(project_id, meta)
    return {"message": f"Added {added_count} papers.", "added": added_count}

# Mode 3
@app.post("/api/projects/{project_id}/evidence-map")
def create_evidence_map(project_id: str, req: EvidenceMapReq):
    engine = ProjectRAGEngine(project_id)
    data = engine.generate_evidence_map(req.topic)
    
    # Save to local project meta
    meta = load_project_meta(project_id)
    if "evidence_maps" not in meta: meta["evidence_maps"] = {}
    meta["evidence_maps"][req.topic] = data
    save_project_meta(project_id, meta)
    
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)