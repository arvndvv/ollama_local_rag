import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import CommandCenter from './components/chat.jsx'

function App() {
  const [count, setCount] = useState(0)

  return (
    <CommandCenter/>
        
  )
}

export default App
