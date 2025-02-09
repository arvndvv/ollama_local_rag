import React, { useState, useEffect, useRef, Fragment, memo, useCallback } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FileText, Link2, Plus, Send, Layers, ArrowRight, 
  Folder, HardDrive, MessageCircle, Settings, MessageSquare, Trash2, Command, Menu, Search, ChevronDown, Loader2, X, Edit2, AlertCircle, FolderPlus
} from 'lucide-react';

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Animated Background Component
const AnimatedBackground = () => (
  <motion.div 
    className="fixed inset-0 z-0 overflow-hidden opacity-50"
    initial={{ opacity: 0 }}
    animate={{ opacity: 0.2 }}
    transition={{ 
      duration: 2, 
      repeat: Infinity, 
      repeatType: "reverse" 
    }}
  >
    <div className="absolute inset-0 bg-gradient-to-br from-[#0a0a0a] via-[#1a2a3a] to-[#0a0a0a] opacity-80"></div>
    <motion.div 
      className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#4a9eff]/10 via-transparent to-transparent"
      animate={{
        backgroundPosition: ['0% 0%', '100% 100%'],
        transition: {
          duration: 10,
          repeat: Infinity,
          repeatType: "reverse"
        }
      }}
    />
  </motion.div>
);

// Glitch Text Effect
const GlitchText = ({ children, className = '' }) => (
  <motion.span
    className={`relative inline-block ${className}`}
    initial={{ opacity: 0.7 }}
    animate={{ 
      opacity: [0.7, 1, 0.7],
      textShadow: [
        '0 0 5px rgba(74,158,255,0.5)',
        '0 0 10px rgba(74,158,255,0.7)',
        '0 0 5px rgba(74,158,255,0.5)'
      ]
    }}
    transition={{
      duration: 3,
      repeat: Infinity,
      repeatType: "reverse"
    }}
  >
    {children}
  </motion.span>
);

// Chat Message Component
const Message = ({ message, timestamp }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
  >
    <div className={`
      max-w-[85%] sm:max-w-[75%] p-4 rounded-2xl
      ${message.role === 'user' 
        ? 'bg-gradient-to-br from-sky-500/20 to-indigo-500/20 ml-4 sm:ml-12' 
        : 'bg-gradient-to-br from-neutral-900/90 to-neutral-900/50 mr-4 sm:mr-12'}
      backdrop-blur-md border border-white/10
    `}>
      <div className="flex items-center gap-2 mb-2">
        <div className={`
          w-2 h-2 rounded-full
          ${message.role === 'user' ? 'bg-sky-500' : 'bg-indigo-500'}
        `} />
        <span className="text-xs font-medium text-white/60">{timestamp}</span>
      </div>
      <p className="text-sm text-white/90 whitespace-pre-wrap">{message.content}</p>
    </div>
  </motion.div>
);

// Chat Input Component
const ChatInput = ({ value, onChange, onSubmit, disabled }) => (
  <div className="p-4 border-t border-white/10">
    <div className="flex items-center gap-2">
      <input
        type="text"
        value={value}
        onChange={onChange}
        onKeyPress={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onSubmit(e);
          }
        }}
        disabled={disabled}
        placeholder="Type your message..."
        className="flex-1 bg-neutral-800/50 border border-white/10 rounded-lg px-4 py-2
          text-white placeholder-white/40 focus:outline-none focus:ring-1 focus:ring-sky-500"
      />
      <button
        onClick={onSubmit}
        disabled={disabled}
        className={`p-2 rounded-lg ${
          disabled 
            ? 'bg-neutral-800/50 text-white/40' 
            : 'bg-sky-500 text-white hover:bg-sky-600'
        }`}
      >
        <Send size={20} />
      </button>
    </div>
  </div>
);

const GradientBackground = () => (
  <div className="fixed inset-0 -z-10">
    <div className="absolute inset-0 bg-[#0a0a0a]" />
    <div className="absolute inset-0 bg-gradient-radial from-indigo-500/10 via-transparent to-transparent" />
    <div className="absolute inset-0 bg-gradient-conic from-sky-500/10 via-transparent to-transparent animate-spin-slow" />
  </div>
);

const Sidebar = memo(({ 
  dropdownRef,
  currentProject,
  projectDocuments,
  setProjectDocuments,
  isAddingDocument,
  setIsAddingDocument,
  documentInput,
  setDocumentInput,
  handleAddDocument,
  handleDeleteDocument,
  isProjectDropdownOpen,
  setIsProjectDropdownOpen,
  searchQuery,
  setSearchQuery,
  filteredProjects,
  setCurrentProject,
  setCurrentChat,
  setChatHistory,
  chats,
  currentChat,
  fetchChatHistory,
  editingChatId,
  setEditingChatId,
  editingChatName,
  setEditingChatName,
  handleRenameChat,
  deleteChat,
  createNewChat,
  linkedProjects,
  setLinkedProjects,
  projects,
  deleteProject,
  createNewProject,
  currentModel,
  setCurrentModel
}) => {
  // Add loading state
  const [isUploading, setIsUploading] = useState(false);
  const [models, setModels] = useState([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  
  // File upload handler
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setIsUploading(true);
      const formData = new FormData();
      formData.append('file', file);
      formData.append('project_name', currentProject);

      await axios.post(
        `${API_BASE_URL}/upload_file`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      // Refresh document list
      const docsResponse = await axios.get(`${API_BASE_URL}/documents`, {
        params: { project_name: currentProject }
      });
      setProjectDocuments(docsResponse.data);
      
      // Clear the file input
      e.target.value = '';
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const ProjectLinkingSection = () => {
    const [isOpen, setIsOpen] = useState(false);

    const handleLinkProjects = async (selectedProjects) => {
      try {
        const response = await axios.post(`${API_BASE_URL}/link_projects/${currentProject}`, {
          linked_projects: selectedProjects
        });
        setLinkedProjects(response.data.linked_projects);
      } catch (error) {
        console.error('Error linking projects:', error);
      }
    };

    return (
      <div className="relative mt-4">
        <div className="mb-2 text-sm text-white/60">Linked Projects</div>
        
        {/* Multiselect Dropdown */}
        <div className="relative">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="w-full px-3 py-2 text-left bg-neutral-800 rounded-lg flex items-center justify-between text-white/80 hover:bg-neutral-700"
          >
            <span>Select Projects</span>
            <ChevronDown size={16} className={`transform transition-transform ${isOpen ? 'rotate-180' : ''}`} />
          </button>

          {/* Dropdown Menu */}
          {isOpen && (
            <div className="absolute z-50 w-full mt-1 bg-neutral-800 rounded-lg shadow-lg max-h-48 overflow-auto">
              {projects
                .filter(p => p !== currentProject)
                .map(project => (
                  <div
                    key={project}
                    onClick={() => {
                      const newSelection = linkedProjects.includes(project)
                        ? linkedProjects.filter(p => p !== project)
                        : [...linkedProjects, project];
                      handleLinkProjects(newSelection);
                    }}
                    className="px-3 py-2 cursor-pointer hover:bg-neutral-700 flex items-center gap-2"
                  >
                    <input
                      type="checkbox"
                      checked={linkedProjects.includes(project)}
                      onChange={() => {}}
                      className="rounded border-white/20 bg-neutral-800"
                    />
                    <span className="text-white/80">{project}</span>
                  </div>
                ))}
            </div>
          )}
        </div>

        {/* Selected Projects Chips */}
        <div className="flex flex-wrap gap-2 mt-2">
          {linkedProjects.map(project => (
            <div
              key={project}
              className="px-2 py-1 rounded-full text-sm flex items-center gap-1 bg-sky-500/20 text-sky-300"
            >
              <span>{project}</span>
              <button
                onClick={() => {
                  const newSelection = linkedProjects.filter(p => p !== project);
                  handleLinkProjects(newSelection);
                }}
                className="hover:text-sky-100"
              >
                <X size={14} />
              </button>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      setIsLoadingModels(true);
      try {
        const response = await axios.get(`${API_BASE_URL}/models`);
        setModels(response.data.models);
      } catch (error) {
        console.error('Error fetching models:', error);
      } finally {
        setIsLoadingModels(false);
      }
    };

    fetchModels();
  }, []);

  return (
    <div className="w-64 h-full bg-neutral-900 border-r border-white/10 flex flex-col">
      <div className="p-4 flex-1 overflow-y-auto">
        {/* Project Selection */}
        <div className="mb-8">
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setIsProjectDropdownOpen(!isProjectDropdownOpen)}
              className="w-full flex items-center justify-between p-2 rounded-lg
                bg-neutral-800/50 text-white hover:bg-neutral-700/50"
            >
              <span>{currentProject}</span>
              <ChevronDown size={16} />
            </button>
            
            {isProjectDropdownOpen && (
              <div className="absolute top-full left-0 w-full mt-1 bg-neutral-800
                rounded-lg shadow-lg overflow-hidden z-50">
                <div className="p-2">
                  <input
                    type="text"
                    placeholder="Search or create new project..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full px-2 py-1 bg-neutral-700 rounded text-white/80
                      placeholder-white/40 outline-none"
                  />
                  {searchQuery && !filteredProjects.includes(searchQuery) && (
                    <button
                      onClick={() => createNewProject(searchQuery)}
                      className="w-full mt-1 p-2 flex items-center gap-2 text-white/60
                        hover:bg-white/5 rounded"
                    >
                      <FolderPlus size={14} />
                      <span>Create "{searchQuery}"</span>
                    </button>
                  )}
                </div>
                <div className="max-h-48 overflow-y-auto">
                  {filteredProjects.map((project) => (
                    <div
                      key={project}
                      className="group flex items-center justify-between hover:bg-white/5"
                    >
                      <button
                        onClick={() => {
                          setCurrentProject(project);
                          setIsProjectDropdownOpen(false);
                          setCurrentChat(null);
                          setChatHistory([]);
                        }}
                        className="flex-1 p-2 text-left text-white/60 hover:text-white"
                      >
                        {project}
                      </button>
                      {project !== 'default' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteProject(project);
                          }}
                          className="p-2 text-white/40 hover:text-red-500
                            opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <Trash2 size={14} />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Linked Projects Section */}
        <div className="px-4 pb-4">
          <ProjectLinkingSection />
        </div>

        {/* Chats Section */}
        <div>
          <h2 className="text-sm font-medium text-white/60 mb-4">Chats</h2>
          <div className="space-y-2">
            {Object.entries(chats[currentProject] || {}).map(([id, chat]) => (
              <div 
                key={id}
                className={`group flex items-center justify-between px-3 py-2
                  rounded-lg bg-neutral-800/50 text-white/60 hover:bg-neutral-700/50 cursor-pointer
                  ${currentChat === id ? 'bg-neutral-700/50' : ''}`}
                onClick={() => {
                  if (editingChatId !== id) {
                    setCurrentChat(id);
                    fetchChatHistory(id);
                  }
                }}
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <MessageCircle size={14} />
                  {editingChatId === id ? (
                    <input
                      type="text"
                      value={editingChatName}
                      onChange={(e) => setEditingChatName(e.target.value)}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                          handleRenameChat(id, editingChatName);
                        }
                      }}
                      onBlur={() => handleRenameChat(id, editingChatName)}
                      className="bg-neutral-700 px-2 py-1 rounded w-full text-white outline-none"
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className="truncate flex-1">
                        {chat.name}
                      </span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingChatId(id);
                          setEditingChatName(chat.name);
                        }}
                        className="opacity-0 group-hover:opacity-100 hover:text-sky-400 transition-opacity duration-200"
                      >
                        <Edit2 size={14} />
                      </button>
                    </div>
                  )}
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteChat(id, e);
                  }}
                  className="opacity-0 group-hover:opacity-100 hover:text-red-500 transition-opacity duration-200 ml-2"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
            <button
              onClick={createNewChat}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg
                border border-white/10 text-white/60 hover:bg-white/5"
            >
              <Plus size={14} />
              <span>New Chat</span>
            </button>
          </div>
        </div>

        {/* Documents Section */}
        <div className="px-3 py-2">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-white/60">Documents</h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsAddingDocument(true)}
                className="text-white/60 hover:text-white"
                title="Add text document"
              >
                <FileText size={16} />
              </button>
              <label className={`cursor-pointer text-white/60 hover:text-white ${isUploading ? 'animate-pulse' : ''}`}>
                <input
                  type="file"
                  className="hidden"
                  accept=".txt,.pdf,.docx"
                  onChange={handleFileUpload}
                  disabled={isUploading}
                />
                {isUploading ? (
                  <div className="w-4 h-4 border-2 border-white/60 border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Plus size={16} />
                )}
              </label>
            </div>
          </div>

          {/* Document List */}
          <div className="space-y-1">
            {projectDocuments.map((doc) => (
              <div
                key={doc.filename}
                className="flex items-center justify-between group px-2 py-1 rounded hover:bg-white/5"
              >
                <span className="text-sm text-white/80 truncate flex items-center gap-2">
                  {doc.filename.endsWith('.txt') ? (
                    <FileText size={14} />
                  ) : doc.filename.endsWith('.pdf') ? (
                    <FileText size={14} />
                  ) : doc.filename.endsWith('.docx') ? (
                    <FileText size={14} />
                  ) : (
                    <FileText size={14} />
                  )}
                  {doc.filename}
                </span>
                <button
                  onClick={() => handleDeleteDocument(doc.filename)}
                  className="text-white/60 hover:text-white opacity-0 group-hover:opacity-100"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>

          {/* Add Document Form */}
          {isAddingDocument && (
            <form onSubmit={handleAddDocument} className="mt-2 space-y-2">
              <input
                type="text"
                placeholder="Filename"
                value={documentInput.filename}
                onChange={(e) => setDocumentInput(prev => ({ ...prev, filename: e.target.value }))}
                className="w-full px-2 py-1 text-sm bg-neutral-800 rounded border border-white/10 focus:border-white/20 focus:outline-none text-white/80"
              />
              <textarea
                placeholder="Content"
                value={documentInput.content}
                onChange={(e) => setDocumentInput(prev => ({ ...prev, content: e.target.value }))}
                className="w-full px-2 py-1 text-sm bg-neutral-800 rounded border border-white/10 focus:border-white/20 focus:outline-none text-white/80 h-24 resize-none"
              />
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => {
                    setIsAddingDocument(false);
                    setDocumentInput({ filename: '', content: '' });
                  }}
                  className="px-2 py-1 text-xs text-white/60 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-2 py-1 text-xs bg-sky-500 text-white rounded hover:bg-sky-600"
                >
                  Add
                </button>
              </div>
            </form>
          )}
        </div>

        {/* Model Selector */}
        <div className="mt-4">
          <h3 className="text-sm font-medium text-white/60 mb-2">Model</h3>
          <select
            value={currentModel}
            onChange={(e) => setCurrentModel(e.target.value)}
            className="w-full bg-neutral-700 text-white border border-white/10 rounded px-2 py-1 text-sm"
            disabled={isLoadingModels}
          >
            {isLoadingModels ? (
              <option>Loading models...</option>
            ) : (
              models.map(model => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))
            )}
          </select>
        </div>
      </div>
    </div>
  );
});

const CommandCenter = () => {
  // State Management
  const [projects, setProjects] = useState(['default']);
  const [currentProject, setCurrentProject] = useState('default');
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Project and Document Management
  const [isProjectDropdownOpen, setIsProjectDropdownOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showLinkModal, setShowLinkModal] = useState(false);
  const [linkedProjects, setLinkedProjects] = useState([]);
  const [documentInput, setDocumentInput] = useState({ filename: '', content: '' });
  const [isAddingDocument, setIsAddingDocument] = useState(false);
  const [projectDocuments, setProjectDocuments] = useState([]);
  
  // Chat Management
  const [chats, setChats] = useState({});
  const [currentChat, setCurrentChat] = useState(null);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editingChatName, setEditingChatName] = useState('');
  const [currentModel, setCurrentModel] = useState('custom-llama'); // Default model

  // Refs
  const chatEndRef = useRef(null);
  const dropdownRef = useRef(null);

  // Filter projects based on search
  const filteredProjects = projects.filter(project => 
    project.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Define fetchProjectDocuments first
  const fetchProjectDocuments = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/documents`, {
        params: { project_name: currentProject }
      });
      setProjectDocuments(response.data || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
      setError('Failed to fetch documents');
    }
  }, [currentProject]);

  // Then use it in handleAddDocument
  const handleAddDocument = useCallback(async (e) => {
    e.preventDefault();
    if (!documentInput.filename.trim() || !documentInput.content.trim()) return;

    try {
      await axios.post(`${API_BASE_URL}/add_text_document`, {
        project_name: currentProject,
        filename: documentInput.filename,
        content: documentInput.content
      });
      
      setDocumentInput({ filename: '', content: '' });
      setIsAddingDocument(false);
      fetchProjectDocuments();
    } catch (error) {
      setError('Failed to add document');
    }
  }, [documentInput, currentProject, fetchProjectDocuments]);

  // Project Management Handlers
  const linkProjects = async () => {
    try {
      await axios.post(`${API_BASE_URL}/link_projects`, null, {
        params: {
          project_name: currentProject,
          linked_projects: linkedProjects
        }
      });
    } catch (error) {
      setError('Failed to link projects');
    }
  };

  const unlinkProject = async (projectToUnlink) => {
    try {
      const updatedLinks = linkedProjects.filter(p => p !== projectToUnlink);
      await axios.post(`${API_BASE_URL}/link_projects`, null, {
        params: {
          project_name: currentProject,
          linked_projects: updatedLinks
        }
      });
      setLinkedProjects(updatedLinks);
    } catch (error) {
      setError('Failed to unlink project');
    }
  };

  // Document Management Handlers
  const handleDeleteDocument = useCallback(async (doc) => {
    try {
      await axios.delete(`${API_BASE_URL}/delete_document`, {
        params: {
          project_name: currentProject,
          filename: doc
        }
      });
      fetchProjectDocuments();
    } catch (error) {
      setError('Failed to delete document');
    }
  }, [currentProject]);

  // Chat Management Handlers
  const createNewChat = async () => {
    const chatId = `chat_${Date.now()}`;
    try {
      await axios.post(`${API_BASE_URL}/create_chat`, null, {
        params: {
          project_name: currentProject,
          chat_id: chatId,
          chat_name: 'New Chat'
        }
      });

      setChats(prev => ({
        ...prev,
        [currentProject]: {
          ...prev[currentProject],
          [chatId]: {
            id: chatId,
            name: 'New Chat',
            messages: []
          }
        }
      }));
      
      setCurrentChat(chatId);
      setChatHistory([]);
    } catch (error) {
      console.error('Error creating chat:', error);
      setError('Failed to create chat');
    }
  };

  const deleteChat = async (chatId, e) => {
    e.stopPropagation();
    try {
      await axios.delete(`${API_BASE_URL}/delete_chat`, {
        params: {
          project_name: currentProject,
          chat_id: chatId
        }
      });

      setChats(prev => {
        const updated = { ...prev };
        if (updated[currentProject]) {
          delete updated[currentProject][chatId];
        }
        return updated;
      });

      if (currentChat === chatId) {
        setCurrentChat(null);
        setChatHistory([]);
      }
    } catch (error) {
      console.error('Error deleting chat:', error);
      setError('Failed to delete chat');
    }
  };

  const handleRenameChat = async (chatId, newName) => {
    try {
      await axios.post(`${API_BASE_URL}/rename_chat`, null, {
        params: {
          project_name: currentProject,
          chat_id: chatId,
          new_name: newName
        }
      });

      setChats(prev => ({
        ...prev,
        [currentProject]: {
          ...prev[currentProject],
          [chatId]: {
            ...prev[currentProject][chatId],
            name: newName
          }
        }
      }));
      setEditingChatId(null);
    } catch (error) {
      console.error('Error renaming chat:', error);
    }
  };

  const fetchChatHistory = useCallback(async (chatId) => {
    if (!chatId || !currentProject) return;
    
    try {
      const response = await axios.get(`${API_BASE_URL}/chat_history`, {
        params: {
          project_name: currentProject,
          chat_id: chatId
        }
      });
      
      setChatHistory(response.data || []);
    } catch (error) {
      console.error('Error fetching chat history:', error);
    }
  }, [currentProject]);

  const initializeProject = useCallback(async (projectName) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/chats`, {
        params: { project_name: projectName }
      });
      
      const projectChats = response.data || {};
      
      if (Object.keys(projectChats).length === 0) {
        const chatId = `chat_${Date.now()}`;
        await axios.post(`${API_BASE_URL}/create_chat`, null, {
          params: {
            project_name: projectName,
            chat_id: chatId,
            chat_name: 'New Chat'
          }
        });
        
        setChats(prev => ({
          ...prev,
          [projectName]: {
            [chatId]: {
              id: chatId,
              name: 'New Chat',
              messages: []
            }
          }
        }));
        
        setCurrentChat(chatId);
        setChatHistory([]);
      } else {
        setChats(prev => ({
          ...prev,
          [projectName]: projectChats
        }));
        
        const firstChatId = Object.keys(projectChats)[0];
        setCurrentChat(firstChatId);
        await fetchChatHistory(firstChatId);
      }
    } catch (error) {
      console.error('Error initializing project:', error);
    }
  }, [fetchChatHistory]);

  useEffect(() => {
    if (currentProject) {
      initializeProject(currentProject);
    }
  }, [currentProject, initializeProject]);

  useEffect(() => {
    if (currentChat) {
      fetchChatHistory(currentChat);
    }
  }, [currentChat, fetchChatHistory]);

  // Fetch all projects
  const fetchProjects = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/projects`);
      setProjects(response.data);
    } catch (error) {
      console.error('Error fetching projects:', error);
    }
  }, []);

  // Auto-name chat based on first message
  const updateChatName = async (chatId, message) => {
    // Take first 30 characters of message and add ellipsis if needed
    const newName = message.length > 30 ? `${message.slice(0, 30)}...` : message;
    
    try {
      await axios.post(`${API_BASE_URL}/rename_chat`, null, {
        params: {
          project_name: currentProject,
          chat_id: chatId,
          new_name: newName
        }
      });

      setChats(prev => ({
        ...prev,
        [currentProject]: {
          ...prev[currentProject],
          [chatId]: {
            ...prev[currentProject][chatId],
            name: newName
          }
        }
      }));
    } catch (error) {
      console.error('Error updating chat name:', error);
    }
  };

  // Update sendMessage to handle auto-naming
  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!chatInput.trim()) return;

    const userMessage = chatInput;
    setChatInput('');
    
    // Immediately update chat history with user message
    setChatHistory(prev => [...prev, { role: 'user', content: userMessage }]);

    // If this is the first message, immediately update chat name
    if (chatHistory.length === 0) {
      const newChatName = userMessage.length > 30 
        ? `${userMessage.slice(0, 30)}...` 
        : userMessage;

      // First call rename_chat API to persist the change
      try {
        await axios.post(`${API_BASE_URL}/rename_chat`, null, {
          params: {
            project_name: currentProject,
            chat_id: currentChat,
            new_name: newChatName
          }
        });

        // Then update local state after successful API call
        setChats(prev => ({
          ...prev,
          [currentProject]: {
            ...prev[currentProject],
            [currentChat]: {
              ...prev[currentProject][currentChat],
              name: newChatName
            }
          }
        }));
      } catch (error) {
        console.error('Error updating chat name:', error);
      }
    }
    
    setIsLoading(true);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        project_name: currentProject,
        chat_id: currentChat,
        content: userMessage,
        model: currentModel  // Add model to request
      });

      // Add AI response to chat history
      setChatHistory(prev => [...prev, { 
        role: 'assistant', 
        content: response.data.response 
      }]);

    } catch (error) {
      console.error('Error sending message:', error);
      setError('Failed to send message');
    } finally {
      setIsLoading(false);
    }
  };

  // Load projects on mount
  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  // Fetch chats when project changes
  useEffect(() => {
    const fetchChats = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/chats`, {
          params: { project_name: currentProject }
        });
        setChats(prev => ({
          ...prev,
          [currentProject]: response.data || {}
        }));
      } catch (error) {
        console.error('Error fetching chats:', error);
        setError('Failed to fetch chats');
      }
    };

    if (currentProject) {
      fetchChats();
      fetchProjectDocuments();
    }
  }, [currentProject]);

  // Scroll to bottom when chat history updates
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsProjectDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Clear project data
  const handleClearProject = async () => {
    if (window.confirm('Are you sure you want to clear all project data?')) {
      try {
        await axios.post(`${API_BASE_URL}/clear_project_data`, null, {
          params: { project_name: currentProject }
        });
        setChatHistory([]);
        setProjectDocuments([]);
        showNotification('Project data cleared successfully');
      } catch (error) {
        showNotification('Failed to clear project data', 'error');
      }
    }
  };

  // Enhanced error handling and toast-like notifications
  const [notification, setNotification] = useState(null);

  const showNotification = (message, type = 'info') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 3000);
  };

  // Fetch linked projects when project changes
  const fetchLinkedProjects = useCallback(async () => {
    if (!currentProject) return;
    
    try {
      const response = await axios.get(`${API_BASE_URL}/projects/${currentProject}`);
      setLinkedProjects(response.data.linked_projects || []);
    } catch (error) {
      console.error('Error fetching linked projects:', error);
      setLinkedProjects([]);
    }
  }, [currentProject]);

  useEffect(() => {
    fetchLinkedProjects();
  }, [currentProject, fetchLinkedProjects]);

  // Add this function in your CommandCenter component
  const deleteProject = async (projectName) => {
    if (!projectName) return;
    
    if (window.confirm(`Are you sure you want to delete project "${projectName}"? This will delete all chats, documents, and context associated with this project.`)) {
        try {
            // Use encodeURIComponent for the project name in the URL
            const encodedProjectName = encodeURIComponent(projectName);
            await axios.delete(`${API_BASE_URL}/project/${encodedProjectName}`);
            
            // Update local state
            setProjects(prev => prev.filter(p => p !== projectName));
            
            // If current project was deleted, switch to another project
            if (currentProject === projectName) {
                const remainingProjects = projects.filter(p => p !== projectName);
                if (remainingProjects.length > 0) {
                    setCurrentProject(remainingProjects[0]);
                    setCurrentChat(null);
                    setChatHistory([]);
                } else {
                    setCurrentProject('default');
                    setCurrentChat(null);
                    setChatHistory([]);
                }
            }
            
            // Clear chats for deleted project
            setChats(prev => {
                const newChats = { ...prev };
                delete newChats[projectName];
                return newChats;
            });
            
        } catch (error) {
            console.error('Error deleting project:', error);
            setError('Failed to delete project');
        }
    }
  };

  // Add this function in your CommandCenter component
  const createNewProject = async (projectName) => {
    try {
      await axios.post(`${API_BASE_URL}/create_project`, null, {
        params: { project_name: projectName }
      });
      
      // Update projects list
      setProjects(prev => [...prev, projectName]);
      
      // Switch to new project
      setCurrentProject(projectName);
      setCurrentChat(null);
      setChatHistory([]);
      setIsProjectDropdownOpen(false);
    } catch (error) {
      console.error('Error creating project:', error);
      setError('Failed to create project');
    }
  };

  return (
    <div className="h-full w-screen flex overflow-hidden bg-neutral-900">
      <GradientBackground />
      
      {/* Link Projects Modal */}
      <AnimatePresence>
        {showLinkModal && <ProjectLinkingModal />}
      </AnimatePresence>

      {/* Sidebar */}
      <div className="hidden md:block">
        <Sidebar
          dropdownRef={dropdownRef}
          currentProject={currentProject}
          projectDocuments={projectDocuments}
          setProjectDocuments={setProjectDocuments}
          isAddingDocument={isAddingDocument}
          setIsAddingDocument={setIsAddingDocument}
          documentInput={documentInput}
          setDocumentInput={setDocumentInput}
          handleAddDocument={handleAddDocument}
          handleDeleteDocument={handleDeleteDocument}
          isProjectDropdownOpen={isProjectDropdownOpen}
          setIsProjectDropdownOpen={setIsProjectDropdownOpen}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          filteredProjects={filteredProjects}
          setCurrentProject={setCurrentProject}
          setCurrentChat={setCurrentChat}
          setChatHistory={setChatHistory}
          chats={chats}
          currentChat={currentChat}
          fetchChatHistory={fetchChatHistory}
          editingChatId={editingChatId}
          setEditingChatId={setEditingChatId}
          editingChatName={editingChatName}
          setEditingChatName={setEditingChatName}
          handleRenameChat={handleRenameChat}
          deleteChat={deleteChat}
          createNewChat={createNewChat}
          linkedProjects={linkedProjects}
          setLinkedProjects={setLinkedProjects}
          projects={projects}
          deleteProject={deleteProject}
          createNewProject={createNewProject}
          currentModel={currentModel}
          setCurrentModel={setCurrentModel}
        />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {currentChat ? (
          <>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {chatHistory.map((message, index) => (
                <MessageComponent 
                  key={index} 
                  message={message} 
                />
              ))}
              {isLoading && (
                <div className="flex items-center space-x-2 p-4 bg-neutral-800/50 rounded-lg">
                  <div className="w-2 h-2 bg-sky-500 rounded-full animate-pulse" />
                  <div className="w-2 h-2 bg-sky-500 rounded-full animate-pulse delay-75" />
                  <div className="w-2 h-2 bg-sky-500 rounded-full animate-pulse delay-150" />
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
            <ChatInput
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onSubmit={handleSubmit}
              disabled={isLoading}
            />
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-white/40">
            Select or create a chat to begin
          </div>
        )}
      </div>
    </div>
  );
};

// Add error notification component
const ErrorNotification = ({ message }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    className="fixed top-4 right-4 bg-red-500/90 text-white px-4 py-2 rounded-lg shadow-lg"
  >
    {message}
  </motion.div>
);

// Update message rendering to show sources
const MessageComponent = ({ message }) => (
  <div className={`flex flex-col ${message.role === 'assistant' ? 'bg-neutral-800/50' : ''} p-4`}>
    <div className="text-white/80 whitespace-pre-wrap">
      {message.content}
    </div>
    {message.sources && message.sources.length > 0 && (
      <div className="mt-2 text-xs text-white/40">
        Sources: {message.sources.join(', ')}
      </div>
    )}
  </div>
);

export default CommandCenter;