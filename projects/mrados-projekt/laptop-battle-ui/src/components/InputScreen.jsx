import { useState } from 'react'

// Laptop SVG icon component
const LaptopIcon = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
    <line x1="2" y1="20" x2="22" y2="20"></line>
  </svg>
)

export default function InputScreen({ onBattle }) {
  const [laptop1, setLaptop1] = useState('')
  const [laptop2, setLaptop2] = useState('')
  const [error, setError] = useState('')

  const handleBattle = () => {
    if (!laptop1.trim() || !laptop2.trim()) {
      setError('Please enter both laptop names')
      return
    }
    if (laptop1.trim().toLowerCase() === laptop2.trim().toLowerCase()) {
      setError('Please enter two different laptops')
      return
    }
    setError('')
    onBattle(laptop1.trim(), laptop2.trim())
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleBattle()
    }
  }

  const isReady = laptop1.trim() && laptop2.trim()

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-5xl md:text-6xl font-extrabold text-white mb-4">
          <span className="text-battle-blue">Laptop</span>
          <span className="text-battle-gold mx-3">Battle</span>
        </h1>
        <p className="text-gray-400 text-lg">
          Reddit Sentiment Showdown - Powered by AI
        </p>
      </div>

      {/* Battle Arena */}
      <div className="flex flex-col md:flex-row items-center gap-6 md:gap-8 w-full max-w-4xl">
        {/* Laptop 1 Input */}
        <div className="flex-1 w-full">
          <div className="glass rounded-2xl p-6 transition-all duration-300 hover:scale-105 border-2 border-transparent hover:border-battle-blue">
            <div className="flex items-center gap-3 mb-4">
              <LaptopIcon className="w-8 h-8 text-battle-blue" />
              <span className="text-battle-blue font-semibold text-lg">Challenger 1</span>
            </div>
            <input
              type="text"
              value={laptop1}
              onChange={(e) => setLaptop1(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="e.g., MacBook Pro M4"
              className="w-full bg-slate-800/50 text-white placeholder-gray-500 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-battle-blue transition-all"
            />
          </div>
        </div>

        {/* VS Divider */}
        <div className="flex-shrink-0">
          <div className="w-20 h-20 rounded-full bg-gradient-to-br from-battle-blue to-battle-red flex items-center justify-center shadow-lg animate-pulse-glow">
            <span className="text-white font-extrabold text-2xl">VS</span>
          </div>
        </div>

        {/* Laptop 2 Input */}
        <div className="flex-1 w-full">
          <div className="glass rounded-2xl p-6 transition-all duration-300 hover:scale-105 border-2 border-transparent hover:border-battle-red">
            <div className="flex items-center gap-3 mb-4">
              <LaptopIcon className="w-8 h-8 text-battle-red" />
              <span className="text-battle-red font-semibold text-lg">Challenger 2</span>
            </div>
            <input
              type="text"
              value={laptop2}
              onChange={(e) => setLaptop2(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="e.g., Lenovo Legion Y540"
              className="w-full bg-slate-800/50 text-white placeholder-gray-500 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-battle-red transition-all"
            />
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <p className="text-red-400 mt-4 text-sm">{error}</p>
      )}

      {/* Battle Button */}
      <button
        onClick={handleBattle}
        disabled={!isReady}
        className={`mt-10 px-12 py-4 rounded-full font-bold text-xl transition-all duration-300 ${
          isReady
            ? 'bg-gradient-to-r from-battle-blue via-purple-500 to-battle-red text-white hover:scale-110 hover:shadow-2xl hover:shadow-purple-500/30 cursor-pointer'
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
        }`}
      >
        Start Battle!
      </button>

      {/* Info text */}
      <p className="text-gray-500 text-sm mt-6 text-center max-w-md">
        We'll scrape Reddit, analyze sentiment using AI, and determine which laptop wins based on real user opinions.
      </p>
    </div>
  )
}
