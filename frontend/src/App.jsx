import { useState } from 'react';
import './App.css'
import CommandCenter from './components/chat.jsx'
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import WritingAssistant from './components/WA.jsx';
import NoteTakingApp from './components/NT.jsx';
function App() {

  return (
    <Router>
      <div className="h-screen w-screen flex flex-col">
        <nav className="bg-neutral-800 p-4 px-8 flex justify-between w-full">
          <div className="flex items-center">
            <span className="text-white text-lg font-bold">notesWiz</span>
          </div>
          <ul className="flex space-x-4">
            <li><a href="/" className="!text-white">Home</a></li>
            <li><a href="/write" className="!text-white">Write</a></li>
          </ul>
        </nav>
      <Routes>
      <Route path="/" element={<CommandCenter/>}/>
      <Route path="/write" element={<NoteTakingApp/>}/>
      </Routes>
      </div>
    </Router>
        
  )
}

export default App
