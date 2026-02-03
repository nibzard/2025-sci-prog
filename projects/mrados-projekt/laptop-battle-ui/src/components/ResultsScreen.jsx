// Trophy icon component
const TrophyIcon = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C13.1 2 14 2.9 14 4V5H16C17.1 5 18 5.9 18 7V8C18 10.21 16.21 12 14 12H13.9C13.5 14.34 11.5 16 9 16H8V18H16V20H8V22H6V20H4V18H6V16H5C2.79 16 1 14.21 1 12V11C1 9.9 1.9 9 3 9H4V4C4 2.9 4.9 2 6 2H12ZM6 4V9H4V11C4 12.1 4.9 13 6 13H5C6.1 13 7 12.1 7 11V4H6ZM14 7H18V8C18 9.1 17.1 10 16 10H14V7ZM8 4H12V9H8V4Z"/>
  </svg>
)

// Check icon
const CheckIcon = () => (
  <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
  </svg>
)

// X icon
const XIcon = () => (
  <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
  </svg>
)

// Score display component
const ScoreDisplay = ({ score, isWinner }) => {
  const getScoreColor = (score) => {
    if (score >= 75) return 'text-green-400'
    if (score >= 50) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <div className={`text-center ${isWinner ? 'animate-winner-glow rounded-xl p-2' : ''}`}>
      <div className={`text-6xl font-extrabold ${getScoreColor(score)}`}>
        {score}
      </div>
      <div className="text-gray-400 text-sm mt-1">Sentiment Score</div>
    </div>
  )
}

// Progress bar component
const ProgressBar = ({ value, color }) => (
  <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
    <div 
      className={`h-full ${color} rounded-full transition-all duration-1000 ease-out progress-bar`}
      style={{ '--progress': `${value}%`, width: `${value}%` }}
    ></div>
  </div>
)

// Laptop card component
const LaptopCard = ({ data, isWinner, color, label }) => {
  const borderColor = isWinner ? 'border-battle-gold' : color === 'blue' ? 'border-battle-blue/30' : 'border-battle-red/30'
  const bgGlow = isWinner ? 'shadow-2xl shadow-battle-gold/30' : ''
  
  return (
    <div className={`glass rounded-2xl p-6 border-2 ${borderColor} ${bgGlow} transition-all duration-500 relative`}>
      {/* Winner badge */}
      {isWinner && (
        <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 bg-battle-gold text-black px-4 py-1 rounded-full font-bold text-sm flex items-center gap-1">
          <TrophyIcon className="w-4 h-4" />
          WINNER
        </div>
      )}
      
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className={`${color === 'blue' ? 'text-battle-blue' : 'text-battle-red'} font-semibold text-sm`}>
          {label}
        </span>
        {data.posts_analyzed > 0 && (
          <span className="text-gray-500 text-xs">
            {data.posts_analyzed} posts analyzed
          </span>
        )}
      </div>

      {/* Laptop Name */}
      <h3 className="text-2xl font-bold text-white mb-4 truncate">
        {data.laptop_name}
      </h3>

      {/* Score */}
      <div className="mb-6">
        <ScoreDisplay score={data.sentiment_score} isWinner={isWinner} />
        <div className="mt-3">
          <ProgressBar 
            value={data.sentiment_score} 
            color={isWinner ? 'bg-battle-gold' : color === 'blue' ? 'bg-battle-blue' : 'bg-battle-red'}
          />
        </div>
      </div>

      {/* Explanation */}
      {data.sentiment_explanation && (
        <p className="text-gray-400 text-sm mb-6 italic">
          "{data.sentiment_explanation}"
        </p>
      )}

      {/* Pros */}
      <div className="mb-4">
        <h4 className="text-green-400 font-semibold mb-2 flex items-center gap-2">
          <CheckIcon /> Pros
        </h4>
        <ul className="space-y-1">
          {data.pros && data.pros.slice(0, 4).map((pro, idx) => (
            <li key={idx} className="text-gray-300 text-sm flex items-start gap-2">
              <span className="text-green-400 mt-1">+</span>
              <span>{pro}</span>
            </li>
          ))}
          {(!data.pros || data.pros.length === 0) && (
            <li className="text-gray-500 text-sm">No pros identified</li>
          )}
        </ul>
      </div>

      {/* Cons */}
      <div className="mb-4">
        <h4 className="text-red-400 font-semibold mb-2 flex items-center gap-2">
          <XIcon /> Cons
        </h4>
        <ul className="space-y-1">
          {data.cons && data.cons.slice(0, 4).map((con, idx) => (
            <li key={idx} className="text-gray-300 text-sm flex items-start gap-2">
              <span className="text-red-400 mt-1">-</span>
              <span>{con}</span>
            </li>
          ))}
          {(!data.cons || data.cons.length === 0) && (
            <li className="text-gray-500 text-sm">No cons identified</li>
          )}
        </ul>
      </div>

      {/* Key Themes */}
      {data.key_themes && data.key_themes.length > 0 && (
        <div className="mt-4">
          <h4 className="text-gray-400 font-semibold mb-2 text-sm">Key Themes</h4>
          <div className="flex flex-wrap gap-2">
            {data.key_themes.slice(0, 3).map((theme, idx) => (
              <span 
                key={idx} 
                className={`px-3 py-1 rounded-full text-xs ${
                  color === 'blue' ? 'bg-battle-blue/20 text-battle-blue' : 'bg-battle-red/20 text-battle-red'
                }`}
              >
                {theme}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Recommendation */}
      {data.user_recommendation && (
        <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
          <p className="text-gray-300 text-sm">
            <span className="font-semibold">Verdict: </span>
            {data.user_recommendation}
          </p>
        </div>
      )}
    </div>
  )
}

export default function ResultsScreen({ data, onReset }) {
  const { laptop1, laptop2, winner, score_difference } = data

  return (
    <div className="min-h-screen p-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl md:text-5xl font-extrabold text-white mb-2">
          Battle Results
        </h1>
        <p className="text-gray-400">
          Based on Reddit sentiment analysis
        </p>
      </div>

      {/* Winner announcement */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-3 bg-battle-gold/20 px-6 py-3 rounded-full">
          <TrophyIcon className="w-8 h-8 text-battle-gold" />
          <span className="text-battle-gold font-bold text-xl">
            {winner === 'tie' 
              ? "It's a Tie!" 
              : `${winner === 'laptop1' ? laptop1.laptop_name : laptop2.laptop_name} Wins!`
            }
          </span>
          {winner !== 'tie' && (
            <span className="text-battle-gold/70 text-sm">
              (+{score_difference} points)
            </span>
          )}
        </div>
      </div>

      {/* Comparison Cards */}
      <div className="grid md:grid-cols-2 gap-6 max-w-5xl mx-auto">
        <LaptopCard 
          data={laptop1} 
          isWinner={winner === 'laptop1'} 
          color="blue"
          label="Challenger 1"
        />
        <LaptopCard 
          data={laptop2} 
          isWinner={winner === 'laptop2'} 
          color="red"
          label="Challenger 2"
        />
      </div>

      {/* Battle Again Button */}
      <div className="text-center mt-10">
        <button
          onClick={onReset}
          className="px-8 py-3 bg-gradient-to-r from-battle-blue to-battle-red text-white font-bold rounded-full hover:scale-105 transition-all duration-300 hover:shadow-xl"
        >
          Battle Again
        </button>
      </div>

      {/* Footer note */}
      <p className="text-center text-gray-600 text-sm mt-8">
        Results based on Reddit user opinions analyzed by AI. Individual experiences may vary.
      </p>
    </div>
  )
}
