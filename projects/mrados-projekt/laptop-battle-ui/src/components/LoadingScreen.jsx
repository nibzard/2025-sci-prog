import { useState, useEffect } from 'react'

// Laptop SVG icon component
const LaptopIcon = ({ className, color }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
    <line x1="2" y1="20" x2="22" y2="20"></line>
  </svg>
)

const loadingMessages = [
  "Searching Reddit for opinions...",
  "Scraping user reviews...",
  "Analyzing sentiment patterns...",
  "Processing with AI...",
  "Comparing specifications...",
  "Calculating battle scores...",
  "Determining the winner...",
]

export default function LoadingScreen({ laptop1, laptop2 }) {
  const [messageIndex, setMessageIndex] = useState(0)
  const [showSparks, setShowSparks] = useState(false)

  useEffect(() => {
    // Rotate through loading messages
    const messageInterval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % loadingMessages.length)
    }, 3000)

    // Show sparks periodically
    const sparkInterval = setInterval(() => {
      setShowSparks(true)
      setTimeout(() => setShowSparks(false), 600)
    }, 2000)

    return () => {
      clearInterval(messageInterval)
      clearInterval(sparkInterval)
    }
  }, [])

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      {/* Header */}
      <h2 className="text-3xl md:text-4xl font-bold text-white mb-12">
        Battle in Progress...
      </h2>

      {/* Battle Animation Container */}
      <div className="relative w-full max-w-2xl h-64 flex items-center justify-center">
        {/* Laptop 1 - Coming from left */}
        <div className="absolute left-0 animate-slide-in-left">
          <div className="flex flex-col items-center">
            <div className="w-24 h-24 md:w-32 md:h-32 bg-battle-blue/20 rounded-2xl flex items-center justify-center animate-battle-shake">
              <LaptopIcon className="w-16 h-16 md:w-20 md:h-20" color="#3b82f6" />
            </div>
            <span className="text-battle-blue font-semibold mt-3 text-sm md:text-base truncate max-w-[120px]">
              {laptop1}
            </span>
          </div>
        </div>

        {/* Center - VS with sparks */}
        <div className="relative z-10">
          <div className={`w-24 h-24 rounded-full bg-gradient-to-br from-battle-blue via-purple-500 to-battle-red flex items-center justify-center shadow-2xl transition-all duration-300 ${
            showSparks ? 'scale-125' : 'scale-100'
          }`}>
            <span className="text-white font-extrabold text-3xl">VS</span>
          </div>
          
          {/* Spark particles */}
          {showSparks && (
            <>
              <div className="spark-particle" style={{ top: '-20px', left: '50%' }}></div>
              <div className="spark-particle" style={{ top: '50%', right: '-20px' }}></div>
              <div className="spark-particle" style={{ bottom: '-20px', left: '50%' }}></div>
              <div className="spark-particle" style={{ top: '50%', left: '-20px' }}></div>
              <div className="spark-particle" style={{ top: '-10px', right: '-10px' }}></div>
              <div className="spark-particle" style={{ bottom: '-10px', left: '-10px' }}></div>
            </>
          )}
        </div>

        {/* Laptop 2 - Coming from right */}
        <div className="absolute right-0 animate-slide-in-right">
          <div className="flex flex-col items-center">
            <div className="w-24 h-24 md:w-32 md:h-32 bg-battle-red/20 rounded-2xl flex items-center justify-center animate-battle-shake">
              <LaptopIcon className="w-16 h-16 md:w-20 md:h-20" color="#ef4444" />
            </div>
            <span className="text-battle-red font-semibold mt-3 text-sm md:text-base truncate max-w-[120px]">
              {laptop2}
            </span>
          </div>
        </div>
      </div>

      {/* Loading Message */}
      <div className="mt-12 text-center">
        <p className="text-gray-400 text-lg transition-all duration-500">
          {loadingMessages[messageIndex]}
        </p>
        
        {/* Loading dots animation */}
        <div className="flex justify-center gap-2 mt-4">
          <div className="w-3 h-3 bg-battle-blue rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-3 h-3 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-3 h-3 bg-battle-red rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-md mt-8">
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-battle-blue via-purple-500 to-battle-red rounded-full animate-pulse" 
               style={{ width: '100%', animation: 'pulse 2s ease-in-out infinite' }}>
          </div>
        </div>
      </div>

      {/* Estimated time note */}
      <p className="text-gray-600 text-sm mt-6">
        First-time analysis may take 1-2 minutes per laptop
      </p>
    </div>
  )
}
