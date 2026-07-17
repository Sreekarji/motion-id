import { useState, useEffect, useRef } from 'react'

export default function ScoreDial({ score, threshold }) {
  const [animatedScore, setAnimatedScore] = useState(0)
  const rafRef = useRef(null)

  // Normalize: cosine sim [-1,1] → display [0,1]
  const displayScore = (score + 1) / 2
  const displayThresh = (threshold + 1) / 2
  const passed = score > threshold

  useEffect(() => {
    let start = null
    const duration = 1200
    const animate = (ts) => {
      if (!start) start = ts
      const progress = Math.min((ts - start) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)  // ease-out cubic
      setAnimatedScore(eased * displayScore)
      if (progress < 1) rafRef.current = requestAnimationFrame(animate)
    }
    rafRef.current = requestAnimationFrame(animate)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [score, threshold])

  const R = 80
  const STROKE = 12
  const CX = 100
  const CY = 95
  // Convert normalized value [0,1] to angle [180, 360]
  const toAngle = (v) => 180 + v * 180
  const toXY = (angleDeg) => {
    const rad = (angleDeg * Math.PI) / 180
    return { x: CX + R * Math.cos(rad), y: CY + R * Math.sin(rad) }
  }

  const arcPath = (startVal, endVal) => {
    const a1 = toAngle(startVal)
    const a2 = toAngle(endVal)
    const p1 = toXY(a1)
    const p2 = toXY(a2)
    const large = (a2 - a1) > 180 ? 1 : 0
    return `M ${p1.x} ${p1.y} A ${R} ${R} 0 ${large} 1 ${p2.x} ${p2.y}`
  }

  const scoreArc = arcPath(0, animatedScore)
  const threshPos = toXY(toAngle(displayThresh))

  return (
    <div style={{ textAlign: 'center' }}>
      <svg width="200" height="120" viewBox="0 0 200 120">
        {/* Background arc */}
        <path d={arcPath(0, 1)} fill="none" stroke="#1E2D4A" strokeWidth={STROKE} strokeLinecap="round" />
        {/* Score arc */}
        <path d={scoreArc} fill="none" stroke={passed ? '#00FF88' : '#FF4444'} strokeWidth={STROKE} strokeLinecap="round" />
        {/* Threshold marker */}
        <line
          x1={threshPos.x - 6} y1={threshPos.y - 6}
          x2={threshPos.x + 6} y2={threshPos.y + 6}
          stroke="#FFD700" strokeWidth="2"
        />
        {/* Center score text */}
        <text x={CX} y={CY - 15} textAnchor="middle" fill="#E8EDF5"
          fontFamily="JetBrains Mono, monospace" fontSize="20" fontWeight="700">
          {score.toFixed(3)}
        </text>
        <text x={CX} y={CY + 2} textAnchor="middle" fill="#6B7E9E"
          fontFamily="JetBrains Mono, monospace" fontSize="10">
          vs threshold {threshold.toFixed(2)}
        </text>
      </svg>
    </div>
  )
}
