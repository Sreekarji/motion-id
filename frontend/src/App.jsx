import { useState, useCallback } from 'react'
import HomePage from './pages/HomePage'
import DemoPage from './pages/DemoPage'
import ResultPage from './pages/ResultPage'

export default function App() {
  const [view, setView] = useState('home')
  const [selectedUser, setSelectedUser] = useState(null)
  const [demoResult, setDemoResult] = useState(null)

  const handleStart = (userId) => {
    setSelectedUser(userId)
    setView('demo')
  }

  const handleComplete = useCallback((result) => {
    setDemoResult(result)
    setView('result')
  }, [])

  const handleReset = () => {
    setSelectedUser(null)
    setDemoResult(null)
    setView('home')
  }

  if (view === 'demo') return <DemoPage userId={selectedUser} onComplete={handleComplete} />
  if (view === 'result') return <ResultPage result={demoResult} onReset={handleReset} />
  return <HomePage onStartDemo={handleStart} />
}
