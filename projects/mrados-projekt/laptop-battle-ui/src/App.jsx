import { useState } from 'react'
import axios from 'axios'
import InputScreen from './components/InputScreen'
import LoadingScreen from './components/LoadingScreen'
import ResultsScreen from './components/ResultsScreen'

// API base URL - uses Vite proxy in development
const API_URL = '/api'

function App() {
  // Stage: "input" | "loading" | "results" | "error"
  const [stage, setStage] = useState('input')
  const [laptop1, setLaptop1] = useState('')
  const [laptop2, setLaptop2] = useState('')
  const [comparisonData, setComparisonData] = useState(null)
  const [error, setError] = useState('')

  const handleBattle = async (l1, l2) => {
    setLaptop1(l1)
    setLaptop2(l2)
    setStage('loading')
    setError('')

    try {
      const response = await axios.post(`${API_URL}/compare`, {
        laptop1: l1,
        laptop2: l2
      }, {
        timeout: 300000 // 5 minute timeout for long analyses
      })

      setComparisonData(response.data)
      setStage('results')
    } catch (err) {
      console.error('Battle failed:', err)
      setError(
        err.response?.data?.detail || 
        err.message || 
        'Something went wrong. Please try again.'
      )
      setStage('error')
    }
  }

  const handleReset = () => {
    setStage('input')
    setLaptop1('')
    setLaptop2('')
    setComparisonData(null)
    setError('')
  }

  return (
    <div className="min-h-screen bg-battle-dark text-white">
      {stage === 'input' && (
        <InputScreen onBattle={handleBattle} />
      )}

      {stage === 'loading' && (
        <LoadingScreen laptop1={laptop1} laptop2={laptop2} />
      )}

      {stage === 'results' && comparisonData && (
        <ResultsScreen data={comparisonData} onReset={handleReset} />
      )}

      {stage === 'error' && (
        <div className="min-h-screen flex flex-col items-center justify-center p-8">
          <div className="glass rounded-2xl p-8 max-w-md text-center">
            <div className="text-6xl mb-4">😵</div>
            <h2 className="text-2xl font-bold text-white mb-4">Battle Failed!</h2>
            <p className="text-gray-400 mb-6">{error}</p>
            <button
              onClick={handleReset}
              className="px-6 py-3 bg-battle-blue text-white font-bold rounded-full hover:scale-105 transition-all"
            >
              Try Again
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
