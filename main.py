import os
import json
import shutil
from typing import List, Dict, Optional
from datetime import datetime
import time
from urllib.parse import unquote  # Add this import at the top

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import chromadb
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
import PyPDF2
import docx
import io
from langchain.schema import Document
from langchain.llms import Ollama
import aiohttp
from langchain.document_loaders import PyPDFLoader  # Add this import at the top

class ProjectManager:
    def __init__(self):
        self.base_path = "./command_center"
        self.projects_file = os.path.join(self.base_path, "projects.json")
        
        # Create base directories
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "storage"), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "documents"), exist_ok=True)
        
        # Initialize with new structure including default chat for each project
        print("Initializing projects.json with default chats")
        default_chat_id = f"chat_{int(time.time())}"  # Use timestamp for default chat ID
        default_projects = {
            "default": {
                "chats": {
                    default_chat_id: {
                        "id": default_chat_id,
                        "name": "New Chat",
                        "messages": []
                    }
                },
                "linked_projects": [],
                "documents": []
            },
            "portloom": {
                "chats": {
                    default_chat_id: {
                        "id": default_chat_id,
                        "name": "New Chat",
                        "messages": []
                    }
                },
                "linked_projects": [],
                "documents": []
            },
            "webarv": {
                "chats": {
                    default_chat_id: {
                        "id": default_chat_id,
                        "name": "New Chat",
                        "messages": []
                    }
                },
                "linked_projects": [],
                "documents": []
            },
            "new project": {
                "chats": {
                    default_chat_id: {
                        "id": default_chat_id,
                        "name": "New Chat",
                        "messages": []
                    }
                },
                "linked_projects": [],
                "documents": []
            }
        }
        
        with open(self.projects_file, 'w') as f:
            json.dump(default_projects, f, indent=2)
        
        # Initialize embeddings and LLM
        self.embeddings = OllamaEmbeddings(model="custom-llama")
        self.llm = OllamaLLM(model="custom-llama")

    def get_project_path(self, project_name):
        return {
            'storage': os.path.join(self.base_path, "storage", project_name),
            'documents': os.path.join(self.base_path, "documents", project_name)
        }

    def create_project(self, project_name):
        with open(self.projects_file, 'r') as f:
            projects = json.load(f)
        
        if project_name in projects:
            return {"status": "Project already exists"}
        
        # Create new project with a fresh chat
        default_chat_id = f"chat_{int(time.time())}"
        projects[project_name] = {
            "chats": {
                default_chat_id: {
                    "id": default_chat_id,
                    "name": "New Chat",
                    "messages": []
                }
            },
            "linked_projects": [],
            "documents": []
        }
        
        with open(self.projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
        
        return {"status": "Project created successfully"}

    def get_project_chats(self, project_name):
        with open(self.projects_file, 'r') as f:
            projects = json.load(f)
            if project_name in projects:
                return projects[project_name]["chats"]
            return {}

    def get_chat_history(self, project_name, chat_id):
        try:
            with open(self.projects_file, 'r') as f:
                projects = json.load(f)
                
            if project_name not in projects:
                return []
                
            if chat_id not in projects[project_name]["chats"]:
                return []
                
            return projects[project_name]["chats"][chat_id]["messages"]
        except Exception as e:
            print(f"Error getting chat history: {str(e)}")  # Debug log
            return []

    def add_document(self, project_name, filename, content):
        # Ensure project exists
        paths = self.get_project_path(project_name)
        file_path = os.path.join(paths['documents'], filename)
        
        # Write content to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Index document
        db = Chroma(
            client=chromadb.PersistentClient(path=paths['storage']),
            embedding_function=self.embeddings
        )
        
        loader = TextLoader(file_path)
        docs = loader.load()
        splits = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
        db.add_documents(splits)
        
        return {"status": "Document added successfully"}

    def get_response(self, project_name, question):
        print(f"Getting response for project: {project_name}")  # Debug
        
        # Prepare context from project and linked projects
        context_docs = []
        with open(self.projects_file, 'r') as f:
            projects = json.load(f)
            target_projects = [project_name] + projects[project_name].get('linked_projects', [])
        
        for proj in target_projects:
            paths = self.get_project_path(proj)
            try:
                db = Chroma(
                    client=chromadb.PersistentClient(path=paths['storage']),
                    embedding_function=self.embeddings
                )
                context_docs.extend(db.similarity_search(question, k=2))
            except Exception as e:
                print(f"Error getting context from {proj}: {str(e)}")
                continue
        
        # Prepare context text
        context_text = "\n".join([doc.page_content for doc in context_docs])
        print(f"Context: {context_text}")
        # Generate response
        prompt = f"""Context: {context_text}

Question: {question}

Please provide a helpful response based on the context if available, or a general response if no context is relevant."""

        try:
            response = self.llm.invoke(prompt)
            return {"response": response}
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {"response": "I apologize, but I encountered an error while processing your request. Please try again."}

    def list_projects(self):
        try:
            print(f"Listing projects from: {self.projects_file}")  # Debug
            
            if not os.path.exists(self.projects_file):
                print("Projects file does not exist!")
                return ["default"]
            
            with open(self.projects_file, 'r') as f:
                content = f.read()
                print(f"Current projects.json content: {content}")  # Debug
                
                projects = json.loads(content)
                project_list = list(projects.keys())
                print(f"Found projects: {project_list}")  # Debug
                
                return project_list
        except Exception as e:
            print(f"Error listing projects: {str(e)}")
            return ["default"]

    def list_documents(self, project_name):
        try:
            paths = self.get_project_path(project_name)
            # Ensure project directories exist
            os.makedirs(paths['documents'], exist_ok=True)
            os.makedirs(paths['storage'], exist_ok=True)
            
            # Check if project exists in projects.json
            with open(self.projects_file, 'r') as f:
                projects = json.load(f)
                if project_name not in projects:
                    return []
            
            # Return list of documents
            return os.listdir(paths['documents'])
        except Exception as e:
            print(f"Error listing documents: {str(e)}")
            return []

    def link_projects(self, project_name, linked_projects):
        with open(self.projects_file, 'r') as f:
            projects = json.load(f)
        
        projects[project_name]['linked_projects'] = linked_projects
        
        with open(self.projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
        
        return {"status": "Projects linked successfully"}

    def create_chat(self, project_name, chat_id, chat_name):
        with open(self.projects_file, 'r') as f:
            projects = json.load(f)
            
        if project_name not in projects:
            return {"error": "Project not found"}
            
        projects[project_name]["chats"][chat_id] = {
            "name": chat_name,
            "messages": []
        }
        
        with open(self.projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
            
        return {"status": "Chat created successfully"}

    def add_message(self, project_name, chat_id, message):
        try:
            with open(self.projects_file, 'r') as f:
                projects = json.load(f)
                
            if project_name not in projects:
                raise ValueError(f"Project {project_name} not found")
                
            if chat_id not in projects[project_name]["chats"]:
                raise ValueError(f"Chat {chat_id} not found in project {project_name}")
            
            # If this is the first message and it's from the user, update chat name
            chat = projects[project_name]["chats"][chat_id]
            if len(chat["messages"]) == 0 and message["role"] == "user":
                # Take first 20 chars or up to the first newline
                title = message["content"][:20].split('\n')[0]
                if len(message["content"]) > 20:
                    title += "..."
                chat["name"] = title
                
            # Add message to the specified chat
            chat["messages"].append(message)
            
            with open(self.projects_file, 'w') as f:
                json.dump(projects, f, indent=2)
                
            return {"status": "Message added successfully", "chat_name": chat["name"]}
        except Exception as e:
            print(f"Error adding message: {str(e)}")
            raise e

    def add_text_document(self, project_name: str, filename: str, content: str):
        paths = self.get_project_path(project_name)
        file_path = os.path.join(paths['documents'], filename)
        
        # Write content to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Update projects file to track documents
        with open(self.projects_file, 'r') as f:
            projects = json.load(f)
        
        if 'documents' not in projects[project_name]:
            projects[project_name]['documents'] = []
        
        projects[project_name]['documents'].append({
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        })
        
        with open(self.projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
        
        # Index document
        return self.add_document(project_name, filename, content)

    def clear_project_data(self, project_name: str):
        paths = self.get_project_path(project_name)
        
        # Clear storage and documents
        shutil.rmtree(paths['storage'], ignore_errors=True)
        shutil.rmtree(paths['documents'], ignore_errors=True)
        os.makedirs(paths['storage'], exist_ok=True)
        os.makedirs(paths['documents'], exist_ok=True)
        
        # Clear chat history and document list
        with open(self.projects_file, 'r') as f:
            projects = json.load(f)
        
        projects[project_name]['chats'] = {}
        projects[project_name]['documents'] = []
        
        with open(self.projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
        
        return {"status": "Project data cleared successfully"}

    def delete_chat(self, project_name, chat_id):
        with open(self.projects_file, 'r') as f:
            projects = json.load(f)
            
        if project_name not in projects or chat_id not in projects[project_name]["chats"]:
            return {"error": "Project or chat not found"}
            
        # If this is the last chat, create a new one before deleting
        if len(projects[project_name]["chats"]) <= 1:
            new_chat_id = f"chat_{int(time.time())}"
            projects[project_name]["chats"][new_chat_id] = {
                "id": new_chat_id,
                "name": "New Chat",
                "messages": []
            }
            
        # Now safe to delete the chat
        del projects[project_name]["chats"][chat_id]
        
        with open(self.projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
            
        return {
            "status": "Chat deleted successfully",
            "new_chat_id": new_chat_id if len(projects[project_name]["chats"]) == 1 else None
        }

    def rename_chat(self, project_name, chat_id, new_name):
        with open(self.projects_file, 'r') as f:
            projects = json.load(f)
            
        if project_name not in projects or chat_id not in projects[project_name]["chats"]:
            raise ValueError("Project or chat not found")
            
        projects[project_name]["chats"][chat_id]["name"] = new_name
        
        with open(self.projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
            
        return {"status": "Chat renamed successfully"}

class LinkProjectsRequest(BaseModel):
    linked_projects: List[str]

class ChatRequest(BaseModel):
    project_name: str
    chat_id: str
    content: str
    model: Optional[str] = "custom-llama"  # Default model if none specified

class DocumentRequest(BaseModel):
    project_name: str
    filename: str
    content: str

# FastAPI Application
app = FastAPI()

# Make CORS more permissive for development
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Project Manager Instance
project_manager = ProjectManager()

# Ollama Client
class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.session = None

    async def ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def chat(self, model: str, messages: list, stream: bool = False):
        session = await self.ensure_session()
        async with session.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": stream
            }
        ) as response:
            return await response.json()

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

# Initialize the Ollama client
ollama_client = OllamaClient()

# API Endpoints
@app.post("/create_project")
async def create_project(project_name: str):
    try:
        projects = safe_read_projects()
        
        if project_name in projects:
            raise HTTPException(status_code=400, detail="Project already exists")
            
        # Initialize new project structure
        projects[project_name] = {
            "chats": {},
            "documents": [],
            "linked_projects": []
        }
        
        # Create project directories
        project_dir = os.path.join("projects", project_name)
        os.makedirs(os.path.join(project_dir, "documents"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "storage"), exist_ok=True)
        
        # Save updated projects.json
        safe_write_projects(projects)
        
        return {"status": "success", "message": f"Project {project_name} created successfully"}
        
    except Exception as e:
        print(f"Error creating project: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_document")
def add_document(project_name: str, filename: str, content: str):
    return project_manager.add_document(project_name, filename, content)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        chat_model = request.model
        embedding_model = "custom-llama"
        print(f"\n=== Starting chat with model: {chat_model} ===")
        
        # Get linked projects
        project_data = await get_project(request.project_name)
        linked_projects = project_data.get("linked_projects", [])
        all_projects = [request.project_name] + linked_projects
        
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Gather context from all projects
        similar_docs = []
        for project in all_projects:
            storage_dir = os.path.join("projects", project, "storage")
            if os.path.exists(storage_dir):
                vectorstore = Chroma(
                    client=chromadb.PersistentClient(path=storage_dir),
                    embedding_function=embeddings
                )
                try:
                    docs = vectorstore.similarity_search(request.content, k=3)
                    similar_docs.extend(docs)
                except Exception as e:
                    print(f"Error getting context from {project}: {str(e)}")
                    continue

        # Build context from all documents
        context_parts = []
        for doc in similar_docs:
            source = doc.metadata.get('source', 'unknown')
            project = doc.metadata.get('project', 'unknown')
            content = doc.page_content
            context_parts.append(f"From {project}/{source}:\n{content}")
        
        context = "\n\n".join(context_parts)
        print(f"Context length: {len(context)}")

        # Get chat history
        chat_dir = os.path.join("projects", request.project_name, "chats")
        chat_file = os.path.join(chat_dir, f"{request.chat_id}.json")
        
        if os.path.exists(chat_file):
            with open(chat_file, 'r') as f:
                chat_data = json.load(f)
        else:
            chat_data = {"id": request.chat_id, "messages": []}

        messages = chat_data.get("messages", [])
        context_messages = messages[-4:] if messages else []

        # Prepare messages with context
        messages = [
            {"role": "system", "content": f"""You are a helpful AI assistant. 
            Use the following relevant excerpts to answer the user's question:
            
            {context}"""},
            {"role": "user", "content": request.content}
        ]

        # Add chat history
        messages[1:1] = context_messages

        # Get response
        response = await ollama_client.chat(
            model=chat_model,
            messages=messages,
            stream=False
        )

        # Update chat history
        chat_data["messages"].extend([
            {"role": "user", "content": request.content},
            {"role": "assistant", "content": response["message"]["content"]}
        ])

        # Save chat
        os.makedirs(chat_dir, exist_ok=True)
        with open(chat_file, 'w') as f:
            json.dump(chat_data, f, indent=2)

        # Get chat name
        chats_file = os.path.join("projects", request.project_name, "chats.json")
        if os.path.exists(chats_file):
            with open(chats_file, 'r') as f:
                chats = json.load(f)
                chat_name = chats.get(request.chat_id, {}).get("name", "New Chat")
        else:
            chat_name = "New Chat"

        return {
            "response": response["message"]["content"],
            "sources": [f"{doc.metadata.get('project', 'unknown')}/{doc.metadata.get('source', 'unknown')}" for doc in similar_docs],
            "chat_name": chat_name
        }

    except Exception as e:
        print(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/projects")
async def get_projects():
    try:
        # Read directly from projects.json in root directory
        projects_file = "projects.json"
        
        if not os.path.exists(projects_file):
            # Create default projects.json if it doesn't exist
            default_data = {
                "default": {
                    "chat_history": [],
                    "linked_projects": []
                }
            }
            with open(projects_file, 'w') as f:
                json.dump(default_data, f, indent=2)
            return ["default"]
            
        with open(projects_file, 'r') as f:
            projects_data = json.load(f)
            
        # Ensure projects directory exists for each project
        for project in projects_data.keys():
            os.makedirs(os.path.join("projects", project), exist_ok=True)
            
        return list(projects_data.keys())
    except Exception as e:
        print(f"Error in get_projects: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents(project_name: str):
    try:
        with open("projects.json", 'r') as f:
            projects = json.load(f)
            
        if project_name not in projects:
            return []
            
        return projects[project_name].get("documents", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/link_projects/{project_name}")
async def link_projects(project_name: str, request: LinkProjectsRequest):
    try:
        projects_file = "projects.json"
        with open(projects_file, 'r') as f:
            projects = json.load(f)
            
        if project_name not in projects:
            raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
            
        projects[project_name]["linked_projects"] = request.linked_projects
        
        with open(projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
            
        return {"status": "success", "linked_projects": request.linked_projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/project_chats")
async def get_project_chats(project_name: str):
    return project_manager.get_project_chats(project_name)

@app.get("/chat_history")
async def get_chat_history(project_name: str, chat_id: str):
    try:
        # Construct path to chat file
        chat_file = os.path.join("projects", project_name, "chats", f"{chat_id}.json")
        
        # If chat file exists, return its messages
        if os.path.exists(chat_file):
            with open(chat_file, 'r') as f:
                chat_data = json.load(f)
                return chat_data.get("messages", [])
        
        return []
        
    except Exception as e:
        print(f"Error getting chat history: {str(e)}")
        return []

@app.post("/create_chat")
async def create_chat(project_name: str, chat_id: str, chat_name: str):
    try:
        project_dir = os.path.join("projects", project_name)
        os.makedirs(project_dir, exist_ok=True)
        
        chats_file = os.path.join(project_dir, "chats.json")
        if os.path.exists(chats_file):
            with open(chats_file, 'r') as f:
                chats = json.load(f)
        else:
            chats = {}
        
        # Ensure we're not overwriting an existing chat
        if chat_id in chats:
            raise HTTPException(status_code=400, detail=f"Chat {chat_id} already exists")
        
        chats[chat_id] = {
            "id": chat_id,
            "name": chat_name,
            "messages": []
        }
        
        with open(chats_file, 'w') as f:
            json.dump(chats, f, indent=2)
            
        return {"status": "success", "chat_id": chat_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_chat_history")
async def update_chat_history(project_name: str, chat_id: str, history: str):
    try:
        project_dir = os.path.join("projects", project_name)
        chats_file = os.path.join(project_dir, "chats.json")
        
        if not os.path.exists(chats_file):
            raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
            
        with open(chats_file, 'r') as f:
            chats = json.load(f)
            
        if chat_id not in chats:
            raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
            
        chats[chat_id]["messages"] = json.loads(history)
            
        with open(chats_file, 'w') as f:
            json.dump(chats, f, indent=2)
            
        return {"status": "success"}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid history format")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rename_chat")
async def rename_chat(
    project_name: str,
    chat_id: str,
    new_name: str
):
    try:
        # Update only chat metadata in chats.json
        chats_file = os.path.join("projects", project_name, "chats.json")
        if os.path.exists(chats_file):
            with open(chats_file, 'r') as f:
                chats = json.load(f)
        else:
            chats = {}
            
        if chat_id not in chats:
            chats[chat_id] = {
                "id": chat_id,
                "name": new_name
            }
        else:
            chats[chat_id]["name"] = new_name
            
        # Save chats.json
        os.makedirs(os.path.dirname(chats_file), exist_ok=True)
        with open(chats_file, 'w') as f:
            json.dump(chats, f, indent=2)
            
        return {"status": "success", "message": "Chat renamed successfully"}
        
    except Exception as e:
        print(f"Error in rename_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats")
async def get_chats(project_name: str):
    try:
        project_dir = os.path.join("projects", project_name)
        os.makedirs(project_dir, exist_ok=True)
        
        chats_file = os.path.join(project_dir, "chats.json")
        if not os.path.exists(chats_file):
            with open(chats_file, 'w') as f:
                json.dump({}, f)
        
        with open(chats_file, 'r') as f:
            chats = json.load(f)
            
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_name}")
async def get_project(project_name: str):
    try:
        projects_file = "projects.json"
        with open(projects_file, 'r') as f:
            projects = json.load(f)
            
        if project_name not in projects:
            raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
            
        return projects[project_name]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_text_document")
async def add_text_document(request: DocumentRequest):
    try:
        project_dir = os.path.join("projects", request.project_name)
        documents_dir = os.path.join(project_dir, "documents")
        storage_dir = os.path.join(project_dir, "storage")
        os.makedirs(documents_dir, exist_ok=True)
        os.makedirs(storage_dir, exist_ok=True)
        
        # Save the document
        filename = request.filename if request.filename.endswith('.txt') else f"{request.filename}.txt"
        file_path = os.path.join(documents_dir, filename)
        with open(file_path, 'w') as f:
            f.write(request.content)

        # Update projects.json
        with open("projects.json", 'r') as f:
            projects = json.load(f)
        if 'documents' not in projects[request.project_name]:
            projects[request.project_name]['documents'] = []
        projects[request.project_name]['documents'].append({
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        })
        with open("projects.json", 'w') as f:
            json.dump(projects, f, indent=2)
         # Get linked projects
        project_data = await get_project(request.project_name)
        linked_projects = project_data.get("linked_projects", [])
        all_projects = [request.project_name] + linked_projects
        # Initialize vector store
        embeddings = OllamaEmbeddings(model="custom-llama")
        # Gather context from all relevant projects
        similar_docs = []
        for project in all_projects:
            storage_dir = os.path.join("projects", project, "storage")
            if os.path.exists(storage_dir):
                vectorstore = Chroma(
                    client=chromadb.PersistentClient(path=storage_dir),
                    embedding_function=embeddings
                )
                try:
                    project_docs = vectorstore.similarity_search(request.content, k=3)
                    similar_docs.extend(project_docs)
                except Exception as e:
                    print(f"Error getting context from {project}: {str(e)}")
                    continue
        # vectorstore = Chroma(
        #     client=chromadb.PersistentClient(path=storage_dir),
        #     embedding_function=embeddings
        # )

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )
        chunks = text_splitter.split_text(request.content)

        # Add chunks to vector store with metadata
        vectorstore.add_texts(
            texts=chunks,
            metadatas=[{
                "source": filename,
                "project": request.project_name,
                "chunk": i
            } for i in range(len(chunks))]
        )

        return {"status": "success"}
    except Exception as e:
        print(f"Error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_chat")
async def delete_chat(project_name: str, chat_id: str):
    try:
        projects_file = "projects.json"
        with open(projects_file, 'r') as f:
            projects = json.load(f)
            
        if project_name not in projects:
            raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
            
        # Get chats for the project
        project_dir = os.path.join("projects", project_name)
        chats_file = os.path.join(project_dir, "chats.json")
        
        if not os.path.exists(chats_file):
            raise HTTPException(status_code=404, detail="Chats file not found")
            
        with open(chats_file, 'r') as f:
            chats = json.load(f)
            
        if chat_id not in chats:
            raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
            
        # Delete the chat
        del chats[chat_id]
        
        # Save updated chats
        with open(chats_file, 'w') as f:
            json.dump(chats, f, indent=2)
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_document")
async def delete_document(project_name: str, filename: str):
    try:
        # Get file paths
        project_dir = os.path.join("projects", project_name)
        documents_dir = os.path.join(project_dir, "documents")
        storage_dir = os.path.join(project_dir, "storage")
        file_path = os.path.join(documents_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document {filename} not found")
        
        # Delete the file
        os.remove(file_path)
        
        # Update projects.json
        projects_file = "projects.json"
        with open(projects_file, 'r') as f:
            projects = json.load(f)
            
        if project_name in projects and 'documents' in projects[project_name]:
            # Remove document from the list
            projects[project_name]['documents'] = [
                doc for doc in projects[project_name]['documents']
                if doc['filename'] != filename
            ]
            
            # Save updated projects.json
            with open(projects_file, 'w') as f:
                json.dump(projects, f, indent=2)
        
        # Delete from ChromaDB
        try:
            db = Chroma(
                client=chromadb.PersistentClient(path=storage_dir),
                embedding_function=OllamaEmbeddings(model="custom-llama")
            )
            
            # Delete all chunks associated with this document
            db.delete(
                where={"source": filename}
            )
        except Exception as e:
            print(f"Error deleting from ChromaDB: {str(e)}")
            # Continue even if ChromaDB deletion fails
        
        return {"status": "success"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/project/{project_name}")
async def delete_project(project_name: str):
    try:
        # Use safe read/write
        projects = safe_read_projects()
        
        if project_name not in projects:
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")
            
        # Delete project directory and all its contents
        project_dir = os.path.join("projects", project_name)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
            
        # Remove project from projects.json
        del projects[project_name]
        
        # Save updated projects.json
        safe_write_projects(projects)
        
        return {"status": "success", "message": f"Project {project_name} deleted successfully"}
        
    except Exception as e:
        print(f"Error deleting project: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    project_name: str = Form(...)
):
    try:
        # Create project directory if it doesn't exist
        project_dir = os.path.join("projects", project_name)
        documents_dir = os.path.join(project_dir, "documents")
        storage_dir = os.path.join(project_dir, "storage")
        os.makedirs(documents_dir, exist_ok=True)
        os.makedirs(storage_dir, exist_ok=True)

        # Save the file
        file_path = os.path.join(documents_dir, file.filename)
        content = await file.read()
        
        # Extract text based on file type
        text_content = ""
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension == '.pdf':
            # Handle PDF files
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    text_parts.append(text)
            text_content = "\n".join(text_parts)
        
        elif file_extension == '.docx':
            # Handle Word documents
            doc_file = io.BytesIO(content)
            doc = docx.Document(doc_file)
            text_content = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        
        elif file_extension == '.txt':
            # Handle text files
            text_content = content.decode('utf-8')
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Save original file
        with open(file_path, "wb") as f:
            f.seek(0)
            f.write(content)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create document chunks
        chunks = text_splitter.create_documents(
            texts=[text_content],
            metadatas=[{
                "source": file.filename,
                "project": project_name
            }]
        )

        # Initialize vector store
        embeddings = OllamaEmbeddings(model="custom-llama")
        vectorstore = Chroma(
            client=chromadb.PersistentClient(path=storage_dir),
            embedding_function=embeddings
        )
        
        # Add chunks to vector store
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Add to vector store with actual content
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas
        )

        # Update projects.json
        projects = safe_read_projects()
        
        if project_name not in projects:
            projects[project_name] = {"documents": [], "chats": {}}
        
        if "documents" not in projects[project_name]:
            projects[project_name]["documents"] = []
            
        projects[project_name]["documents"].append({
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        })
        
        safe_write_projects(projects)

        return {
            "status": "success",
            "message": f"File {file.filename} uploaded and processed successfully",
            "chunks_created": len(chunks)
        }

    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize projects.json if it doesn't exist or is corrupted"""
    try:
        # Try to read projects.json
        with open("projects.json", 'r') as f:
            projects = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is corrupted, create with default structure
        default_structure = {
            "default": {
                "chats": {},
                "documents": [],
                "linked_projects": []
            }
        }
        
        # Create projects directory if it doesn't exist
        os.makedirs("projects", exist_ok=True)
        
        # Write default structure
        with open("projects.json", 'w') as f:
            json.dump(default_structure, f, indent=2)
            
        print("Initialized new projects.json with default structure")

def safe_read_projects():
    """Safely read projects.json with backup recovery"""
    try:
        with open("projects.json", 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Try to read backup if it exists
        try:
            with open("projects.json.backup", 'r') as f:
                projects = json.load(f)
            # Restore from backup
            with open("projects.json", 'w') as f:
                json.dump(projects, f, indent=2)
            return projects
        except (FileNotFoundError, json.JSONDecodeError):
            # Create new default structure
            default_structure = {
                "default": {
                    "chats": {},
                    "documents": [],
                    "linked_projects": []
                }
            }
            with open("projects.json", 'w') as f:
                json.dump(default_structure, f, indent=2)
            return default_structure

def safe_write_projects(projects):
    """Safely write projects.json with backup"""
    try:
        # Create backup of current file if it exists
        if os.path.exists("projects.json"):
            shutil.copy2("projects.json", "projects.json.bak")
            
        # Write new content
        with open("projects.json", "w") as f:
            json.dump(projects, f, indent=2)
            
        print("Successfully wrote to projects.json")
        # Log the content that was written for debugging
        with open("projects.json", "r") as f:
            content = json.load(f)
            print(f"Projects structure after save: {json.dumps(content, indent=2)}")
            
    except Exception as e:
        print(f"Error writing projects.json: {str(e)}")
        if os.path.exists("projects.json.bak"):
            shutil.copy2("projects.json.bak", "projects.json")
            print("Restored from backup")
        raise

# Add cleanup on app shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await ollama_client.close()

@app.get("/models")
async def get_models():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract model names from the response
                    models = [model["name"] for model in data["models"]]
                    return {"models": models}
                else:
                    raise HTTPException(status_code=response.status, detail="Failed to fetch models from Ollama")
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex/{project_name}")
async def reindex_project(project_name: str):
    try:
        print(f"\n=== Starting reindex for project: {project_name} ===")
        
        base_dir = os.path.join("projects", project_name)
        documents_dir = os.path.join(base_dir, "documents")
        
        print(f"\nChecking directories:")
        print(f"Base directory: {base_dir}")
        print(f"Documents directory: {documents_dir}")
        
        if os.path.exists(documents_dir):
            print("\nListing documents directory contents:")
            files = os.listdir(documents_dir)
            print(f"Found files: {files}")
            
            embedding_model = "custom-llama"
            storage_dir = os.path.join(base_dir, "storage", embedding_model)
            
            if os.path.exists(storage_dir):
                print(f"\nClearing existing storage: {storage_dir}")
                shutil.rmtree(storage_dir)
            os.makedirs(storage_dir, exist_ok=True)
            
            embeddings = OllamaEmbeddings(model=embedding_model)
            vectorstore = Chroma(
                client=chromadb.PersistentClient(path=storage_dir),
                embedding_function=embeddings
            )
            
            doc_count = 0
            for filename in files:
                file_path = os.path.join(documents_dir, filename)
                print(f"\nProcessing file: {file_path}")
                
                if os.path.isfile(file_path):
                    try:
                        print(f"Reading: {filename}")
                        
                        # Handle different file types
                        if filename.lower().endswith('.pdf'):
                            # Use PyPDFLoader for PDF files
                            loader = PyPDFLoader(file_path)
                            pages = loader.load()
                            content = "\n\n".join(page.page_content for page in pages)
                            print(f"Read PDF content length: {len(content)}")
                        else:
                            # Use regular text loading for other files
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                print(f"Read text content length: {len(content)}")
                        
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        chunks = text_splitter.create_documents(
                            texts=[content],
                            metadatas=[{"source": filename, "project": project_name}]
                        )
                        
                        texts = [chunk.page_content for chunk in chunks]
                        metadatas = [chunk.metadata for chunk in chunks]
                        vectorstore.add_texts(texts=texts, metadatas=metadatas)
                        print(f"Successfully indexed {filename} - {len(chunks)} chunks")
                        doc_count += 1
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
            
            print(f"\nSuccessfully indexed {doc_count} documents")
            return {"status": "success", "message": f"Reindexed {doc_count} documents"}
        else:
            return {"status": "error", "message": "Documents directory not found"}
            
    except Exception as e:
        print(f"Error during reindex: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)