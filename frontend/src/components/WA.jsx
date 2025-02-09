import React, { useState, useRef, useEffect } from 'react';
import { Command } from 'lucide-react';
import axios from 'axios';

const WritingAssistant = () => {
  const [content, setContent] = useState('');
  const [suggestion, setSuggestion] = useState('');
  const [cursorPosition, setCursorPosition] = useState(0);
  const [isFetching, setIsFetching] = useState(false);
  const [typingTimeout, setTypingTimeout] = useState(null);
  
  // Fixed ref declarations
  const hiddenTextareaRef = useRef(null);
  const visibleTextRef = useRef(null);
  
  // Function to get suggestions from the LLM
  const getSuggestion = async (text, cursorPos) => {
    // Only suggest if there are at least a few words
    const wordsBeforeCursor = text.slice(0, cursorPos).trim().split(/\s+/).length;
    if (wordsBeforeCursor < 3) {
      setSuggestion('');
      return;
    }
    
    try {
      setIsFetching(true);
      const response = await axios.post('http://localhost:8000/chat', {
        project_name: 'default',
        chat_id: 'suggestions',
        content: `Continue this text naturally (respond with continuation only, no explanations): ${text.slice(0, cursorPos)}`,
        model: 'custom-llama'
      });
      
      if (response.data.response) {
        setSuggestion(response.data.response.split('\n')[0]); // Take only first line
      }
    } catch (error) {
      console.error('Error getting suggestion:', error);
    } finally {
      setIsFetching(false);
    }
  };

  // Handle text changes
  const handleTextChange = (e) => {
    const newContent = e.target.value;
    const newPosition = e.target.selectionStart;
    
    setContent(newContent);
    setCursorPosition(newPosition);
    setSuggestion(''); // Clear previous suggestion
    
    // Clear existing timeout
    if (typingTimeout) {
      clearTimeout(typingTimeout);
    }
    
    // Set new timeout for 3 seconds of inactivity
    const newTimeout = setTimeout(() => {
      getSuggestion(newContent, newPosition);
    }, 3000);
    
    setTypingTimeout(newTimeout);
  };

  // Handle cursor position updates
  const handleSelect = (e) => {
    setCursorPosition(e.target.selectionStart);
  };

  // Handle keyboard shortcuts
  const handleKeyDown = (e) => {
    // Tab to accept suggestion
    if (e.key === 'Tab' && suggestion) {
      e.preventDefault();
      const beforeCursor = content.slice(0, cursorPosition);
      const afterCursor = content.slice(cursorPosition);
      const newContent = beforeCursor + suggestion + afterCursor;
      
      setContent(newContent);
      setCursorPosition(cursorPosition + suggestion.length);
      setSuggestion('');
    }
    
    // "/" to get new suggestion when one is already shown
    if (e.key === '/' && suggestion) {
      e.preventDefault();
      getSuggestion(content, cursorPosition);
    }
  };

  // Split content into parts for rendering
  const getContentParts = () => {
    if (!suggestion) return [content];
    
    const beforeCursor = content.slice(0, cursorPosition);
    const afterCursor = content.slice(cursorPosition);
    return [beforeCursor, suggestion, afterCursor];
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (typingTimeout) {
        clearTimeout(typingTimeout);
      }
    };
  }, [typingTimeout]);

  // Keep hidden textarea in sync with visible text
  useEffect(() => {
    if (hiddenTextareaRef.current) {
      hiddenTextareaRef.current.selectionStart = cursorPosition;
      hiddenTextareaRef.current.selectionEnd = cursorPosition;
    }
  }, [cursorPosition]);

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <div className="relative font-mono text-base leading-relaxed">
        {/* Hidden textarea for actual input */}
        <textarea
          ref={hiddenTextareaRef}
          value={content}
          onChange={handleTextChange}
          onKeyDown={handleKeyDown}
          onSelect={handleSelect}
          className="absolute inset-0 opacity-0 z-10 h-96 w-full resize-none"
          autoFocus
        />
        
        {/* Visible text display */}
        <div 
          ref={visibleTextRef}
          className="w-full h-96 p-4 bg-neutral-800 text-white rounded-lg
            border border-neutral-700 focus-within:border-sky-500 whitespace-pre-wrap"
        >
          {getContentParts().map((part, index) => {
            // Index 1 is always the suggestion if it exists
            if (index === 1 && suggestion) {
              return (
                <span key={index} className="text-neutral-500">
                  {part}
                </span>
              );
            }
            return <span key={index}>{part}</span>;
          })}
          {!content && (
            <span className="text-neutral-500">
              Start writing... (Tab to accept suggestions, "/" for new ones)
            </span>
          )}
        </div>
        
        {/* Loading indicator */}
        {isFetching && (
          <div className="absolute right-4 top-4 text-sky-500">
            <div className="animate-spin w-4 h-4 border-2 border-current border-t-transparent rounded-full" />
          </div>
        )}
      </div>
      
      {/* Shortcuts info */}
      <div className="mt-4 text-neutral-400 text-sm flex items-center gap-2">
        <Command size={16} />
        <span>Tab to accept suggestion • "/" for new suggestion when one is shown</span>
      </div>
    </div>
  );
};

export default WritingAssistant;