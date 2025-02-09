import React, { useState, useRef, useEffect } from 'react';
import { Plus, Save, Trash2, Search, Settings, ChevronDown, Edit2, Check, Menu } from 'lucide-react';
import axios from 'axios';

// API configuration
const API_BASE_URL = 'http://localhost:8001';

const NoteTakingApp = () => {
  const [notes, setNotes] = useState([]);
  const [activeNote, setActiveNote] = useState(null);
  const [content, setContent] = useState('');
  const [suggestion, setSuggestion] = useState('');
  const [cursorPosition, setCursorPosition] = useState(0);
  const [isFetching, setIsFetching] = useState(false);
  const [typingTimeout, setTypingTimeout] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState('');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  const editorRef = useRef(null);

  // Load notes on mount
  useEffect(() => {
    fetchNotes();
  }, []);

  const fetchNotes = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/notes`);
      setNotes(response.data);
      if (response.data.length > 0 && !activeNote) {
        loadNote(response.data[0].id);
      }
    } catch (error) {
      console.error('Error loading notes:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadNote = async (noteId) => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${API_BASE_URL}/notes/${noteId}`);
      setActiveNote(response.data);
      setContent(response.data.content);
      editorRef.current?.focus();
    } catch (error) {
      console.error('Error loading note:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const saveNote = async () => {
    if (!activeNote) return;
    
    try {
      const response = await axios.put(`${API_BASE_URL}/notes/${activeNote.id}`, {
        content: content,
        title: activeNote.title
      });
      setActiveNote(response.data);
      fetchNotes(); // Refresh list to update timestamps
    } catch (error) {
      console.error('Error saving note:', error);
    }
  };

  const createNewNote = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/notes`, {
        title: "Untitled Note"
      });
      await fetchNotes();
      loadNote(response.data.id);
    } catch (error) {
      console.error('Error creating note:', error);
    }
  };

  const deleteNote = async (noteId) => {
    if (!window.confirm('Are you sure you want to delete this note?')) return;
    
    try {
      await axios.delete(`${API_BASE_URL}/notes/${noteId}`);
      await fetchNotes();
      if (activeNote?.id === noteId) {
        setActiveNote(null);
        setContent('');
      }
    } catch (error) {
      console.error('Error deleting note:', error);
    }
  };

  const updateNoteTitle = async () => {
    if (!activeNote || !editTitle.trim()) return;
    
    try {
      const response = await axios.put(`${API_BASE_URL}/notes/${activeNote.id}`, {
        title: editTitle,
        content: activeNote.content
      });
      setActiveNote(response.data);
      fetchNotes();
      setIsEditing(false);
    } catch (error) {
      console.error('Error updating note title:', error);
    }
  };

  const getSuggestion = async () => {
    if (!activeNote || !content || !editorRef.current) return;
    
    try {
      setIsFetching(true);
      const cursorPos = editorRef.current.selectionStart;
      
      const response = await axios.post(`${API_BASE_URL}/notes/${activeNote.id}/suggest`, {
        current_text: content,
        cursor_position: cursorPos
      });
      
      if (response.data.suggestion) {
        console.log('Received suggestion:', response.data.suggestion); // Debug log
        setSuggestion(response.data.suggestion);
      }
    } catch (error) {
      console.error('Error getting suggestion:', error);
      setSuggestion('');
    } finally {
      setIsFetching(false);
    }
  };

  const handleTextChange = (e) => {
    const newContent = e.target.value;
    setContent(newContent);
    
    // Clear existing timeout and suggestion
    if (typingTimeout) {
      clearTimeout(typingTimeout);
    }
    setSuggestion('');
    
    // Set new timeout for suggestions
    const newTimeout = setTimeout(() => {
      if (editorRef.current) {
        setCursorPosition(editorRef.current.selectionStart);
        getSuggestion();
      }
    }, 3000);
    
    setTypingTimeout(newTimeout);
  };
  const handleSelectionChange = () => {
    if (editorRef.current) {
      setCursorPosition(editorRef.current.selectionStart);
    }
  };
  const handleClick = () => {
    if (editorRef.current) {
      setCursorPosition(editorRef.current.selectionStart);
    }
  };

  // Modified keydown handler
  const handleKeyDown = (e) => {
    // Save with Ctrl/Cmd + S
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault();
      saveNote();
      return;
    }
    
    // New note with Ctrl/Cmd + N
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
      e.preventDefault();
      createNewNote();
      return;
    }
    
    // Tab to accept suggestion
    if (e.key === 'Tab' && suggestion) {
      e.preventDefault();
      const textArea = editorRef.current;
      if (!textArea) return;

      const cursorPos = textArea.selectionStart;
      const beforeCursor = content.slice(0, cursorPos);
      const afterCursor = content.slice(cursorPos);
      const newContent = beforeCursor + suggestion + afterCursor;
      
      setContent(newContent);
      
      // Update cursor position after React re-renders
      setTimeout(() => {
        if (textArea) {
          const newPosition = cursorPos + suggestion.length;
          textArea.selectionStart = newPosition;
          textArea.selectionEnd = newPosition;
          setCursorPosition(newPosition);
        }
      }, 0);
      
      setSuggestion('');
    }
    
    // "/" to get new suggestion when one is already shown
    if (e.key === '/' && suggestion) {
      e.preventDefault();
      getSuggestion();
    }
  };
  useEffect(() => {
    if (editorRef.current) {
      editorRef.current.selectionStart = cursorPosition;
      editorRef.current.selectionEnd = cursorPosition;
    }
  }, [cursorPosition]);

  const filteredNotes = notes.filter(note => 
    note.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    note.content.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="h-screen flex bg-white dark:bg-neutral-900">
      {/* Mobile menu button */}
      <button
        className="md:hidden fixed top-4 left-4 z-50 p-2 rounded-lg bg-neutral-100 dark:bg-neutral-800"
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
      >
        <Menu size={20} />
      </button>

      {/* Sidebar */}
      <div className={`
        fixed md:static inset-y-0 left-0 z-40 w-80 transform 
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        md:translate-x-0 transition-transform duration-200 ease-in-out
        bg-white dark:bg-neutral-900 border-r border-neutral-200 dark:border-neutral-800 
        flex flex-col
      `}>
        {/* Sidebar Header */}
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-800">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-semibold text-neutral-800 dark:text-white">Notes</h1>
            <button
              onClick={createNewNote}
              className="p-2 text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 
                dark:hover:text-white rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800
                transition-colors duration-200"
              title="New Note (Ctrl/Cmd + N)"
            >
              <Plus size={20} />
            </button>
          </div>
          
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-400" size={16} />
            <input
              type="text"
              placeholder="Search notes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-neutral-100 dark:bg-neutral-800 rounded-lg
                text-neutral-900 dark:text-neutral-100 placeholder-neutral-500 dark:placeholder-neutral-400
                focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200"
            />
          </div>
        </div>
        
        {/* Notes List */}
        <div className="flex-1 overflow-y-auto">
          {filteredNotes.map((note) => (
            <div
              key={note.id}
              className={`group p-4 border-b border-neutral-200 dark:border-neutral-800 cursor-pointer
                transition-colors duration-200
                ${activeNote?.id === note.id 
                  ? 'bg-blue-50 dark:bg-blue-900/20' 
                  : 'hover:bg-neutral-50 dark:hover:bg-neutral-800/50'}`}
              onClick={() => loadNote(note.id)}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-neutral-900 dark:text-white truncate">
                  {note.title}
                </h3>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteNote(note.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 text-neutral-400 
                    hover:text-red-500 rounded transition-opacity duration-200"
                >
                  <Trash2 size={16} />
                </button>
              </div>
              <p className="text-sm text-neutral-500 dark:text-neutral-400 truncate">
                {note.content || 'No content'}
              </p>
              <p className="text-xs text-neutral-400 dark:text-neutral-500 mt-2">
                {new Date(note.updated_at).toLocaleString()}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        {activeNote && (
          <div className="border-b border-neutral-200 dark:border-neutral-800 p-4 flex items-center justify-between">
            <div className="flex items-center gap-2 flex-1">
              {isEditing ? (
                <div className="flex items-center gap-2 flex-1">
                  <input
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    className="flex-1 px-2 py-1 bg-neutral-100 dark:bg-neutral-800 rounded
                      text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 
                      focus:ring-blue-500"
                    autoFocus
                  />
                  <button
                    onClick={updateNoteTitle}
                    className="p-1 text-green-500 hover:text-green-600 rounded"
                  >
                    <Check size={20} />
                  </button>
                </div>
              ) : (
                <>
                  <h2 className="text-lg font-medium text-neutral-900 dark:text-white">
                    {activeNote.title}
                  </h2>
                  <button
                    onClick={() => {
                      setIsEditing(true);
                      setEditTitle(activeNote.title);
                    }}
                    className="p-1 text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300 rounded"
                  >
                    <Edit2 size={16} />
                  </button>
                </>
              )}
            </div>
            <button
              onClick={saveNote}
              className="p-2 text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300 rounded"
              title="Save (Ctrl/Cmd + S)"
            >
              <Save size={20} />
            </button>
          </div>
        )}

       {/* Editor */}
<div className="flex-1 overflow-hidden p-4">
  <div className="relative h-full font-mono">
    <div className="w-full h-full p-4 bg-white dark:bg-neutral-900 
      text-neutral-900 dark:text-neutral-100 rounded-lg
      border dark:border-neutral-800 whitespace-pre-wrap"
    >
      {/* Content before cursor */}
      <span>{content.slice(0, cursorPosition)}</span>
      
      {/* Suggestion */}
      {suggestion && !isFetching && (
        <span className="text-neutral-500 dark:text-neutral-400">
          {suggestion}
        </span>
      )}
      
      {/* Content after cursor */}
      <span>{content.slice(cursorPosition)}</span>

      {/* Hidden textarea for input handling */}
      <textarea
        ref={editorRef}
        value={content}
        onChange={handleTextChange}
        onKeyDown={handleKeyDown}
        onClick={handleClick}
        onSelect={handleSelectionChange}
        disabled={!activeNote || isLoading}
        spellCheck={false}
        className="absolute inset-0 w-full h-full p-4 bg-transparent
          text-neutral-900 dark:text-neutral-100 resize-none focus:outline-none
          placeholder-neutral-400 dark:placeholder-neutral-600 text-base leading-relaxed
          caret-blue-500"
        style={{ 
          WebkitTextFillColor: 'transparent',
          caretColor: 'rgb(59, 130, 246)'
        }}
        placeholder={activeNote ? "Start writing... (Tab to accept suggestions, '/' for new ones)" : "Select or create a note to start writing"}
      />
    </div>

    {/* Loading indicator */}
    {isFetching && (
      <div className="absolute right-6 top-6 flex items-center gap-2">
        <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full" />
        <span className="text-sm text-neutral-400">Generating...</span>
      </div>
    )}
  </div>
</div>
</div></div>

  );
};

export default NoteTakingApp;