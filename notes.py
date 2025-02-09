from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

# Create a new FastAPI app instance for notes
notes_app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
]

notes_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class NoteCreate(BaseModel):
    title: str = "Untitled Note"

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

class NoteResponse(BaseModel):
    id: str
    title: str
    content: str
    created_at: str
    updated_at: str

class SuggestionRequest(BaseModel):
    current_text: str
    cursor_position: int

# Initialize notes directory
NOTES_DIR = "./notes"
os.makedirs(NOTES_DIR, exist_ok=True)

# Note management endpoints
@notes_app.get("/notes", response_model=List[NoteResponse])
async def get_notes():
    """Get all notes metadata"""
    try:
        notes = []
        for filename in os.listdir(NOTES_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(NOTES_DIR, filename), 'r') as f:
                    note_data = json.load(f)
                    notes.append(note_data)
        return sorted(notes, key=lambda x: x['updated_at'], reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@notes_app.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(note_id: str):
    """Get a specific note by ID"""
    try:
        file_path = os.path.join(NOTES_DIR, f"{note_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Note not found")
            
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@notes_app.post("/notes", response_model=NoteResponse)
async def create_note(note: NoteCreate):
    """Create a new note"""
    try:
        note_id = f"note_{int(datetime.now().timestamp())}"
        note_data = {
            "id": note_id,
            "title": note.title,
            "content": "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        file_path = os.path.join(NOTES_DIR, f"{note_id}.json")
        with open(file_path, 'w') as f:
            json.dump(note_data, f, indent=2)
            
        return note_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@notes_app.put("/notes/{note_id}", response_model=NoteResponse)
async def update_note(note_id: str, note: NoteUpdate):
    """Update a note's content and/or title"""
    try:
        file_path = os.path.join(NOTES_DIR, f"{note_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Note not found")
            
        with open(file_path, 'r') as f:
            note_data = json.load(f)
            
        if note.title is not None:
            note_data['title'] = note.title
        if note.content is not None:
            note_data['content'] = note.content
            
        note_data['updated_at'] = datetime.now().isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(note_data, f, indent=2)
            
        return note_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@notes_app.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete a note"""
    try:
        file_path = os.path.join(NOTES_DIR, f"{note_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Note not found")
            
        os.remove(file_path)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@notes_app.post("/notes/{note_id}/suggest")
async def get_suggestion(note_id: str, request: SuggestionRequest):
    """Get AI suggestion for note content"""
    try:
        from langchain_ollama import OllamaLLM
        
        llm = OllamaLLM(model="custom-llama")
        
        prompt = f"""Given this note context, suggest a natural continuation that matches the style and content:
        ###
        {request.current_text[:request.cursor_position]}
        ###
        Consider:
        - The topic and theme of the current text
        - The writing style (formal/informal)
        - Any lists or patterns in the text
        - The current paragraph's context
        
        Respond with ONLY the continuation text, no explanations."""
        
        suggestion = llm.invoke(prompt).split('\n')[0]
        return {"suggestion": suggestion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For running the app directly (development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(notes_app, host="0.0.0.0", port=8001)  # Note: Using port 8001 to avoid conflict with main app