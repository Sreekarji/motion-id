import { useState, useEffect, useRef } from 'react'
import { runDemo } from '../api'
import SensorChart from '../components/SensorChart'
import ScoreDial from '../components/ScoreDial'

const SENSORS = ['ACC', 'GRAV', 'GYRO', 'LIN', 'MAG', 'ROT']

function LoadingPhase() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', padding: 40 }}>
      <p className="mono text-cyan" style={{ fontSize: 18, marginBottom: 32 }}>
        Analyzing motion pattern...
      </p>
      {/* Fake waveforms */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, maxWidth: 600, width: '100%', marginBottom: 32 }}>
        {SENSORS.map((s, i) => (
          <div key={s} className="card" style={{ padding: 12, textAlign: 'center' }}>
            <p className="mono" style={{ fontSize: 10, color: '#6B7E9E', marginBottom: 8 }}>{s}</p>
            <svg width="100%" height="40" viewBox="0 0 160 40">
              <path
                d={`M0,20 ${Array.from({length: 32}, (_, j) =>
                  `Q${j*5+2.5},${20 + Math.sin((j + i*3) * 0.8) * 15} ${(j+1)*5},20`
                ).join(' ')}`}
                fill="none" stroke="#00D4FF" strokeWidth="1.5" opacity="0.6"
              >
                <animate attributeName="stroke-dashoffset" from="320" to="0" dur={`${1.5 + i*0.2}s`} repeatCount="indefinite" />
                <animate attributeName="stroke-dasharray" values="0 320;160 160;320 0" dur={`${1.5 + i*0.2}s`} repeatCount="indefinite" />
              </path>
            </svg>
          </div>
        ))}
      </div>
      {/* Progress bar */}
      <div style={{ width: 300, height: 4, background: '#1E2D4A', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ width: '100%', height: '100%', background: '#00D4FF', borderRadius: 2, animation: 'loadBar 2s ease-in-out' }} />
      </div>
      <style>{`
        @keyframes loadBar { from { transform: translateX(-100%); } to { transform: translateX(0); } }
      `}</style>
    </div>
  )
}

function MPIPhase({ mpi, features, onSkip }) {
  const isBypassed = mpi.note && mpi.note.includes('bypassed')
  const isUnlock = mpi.is_unlock

  let cardColor, title, subtitle
  if (isBypassed) {
    cardColor = '#00D4FF'; title = 'MPI STAGE: DEMO MODE'; subtitle = 'Proceeding to identity verification'
  } else if (isUnlock) {
    cardColor = '#00FF88'; title = 'UNLOCK PATTERN DETECTED'; subtitle = `Confidence: ${(mpi.confidence * 100).toFixed(1)}%`
  } else {
    cardColor = '#FF4444'; title = 'NO UNLOCK PATTERN'; subtitle = 'Authentication rejected at MPI stage'
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 40, position: 'relative' }}>
      <button className="btn-ghost" onClick={onSkip} style={{ position: 'absolute', top: 20, right: 20, fontSize: 12, padding: '6px 16px' }}>
        Skip →
      </button>

      {/* Sensor chart */}
      {features && (
        <div style={{ marginBottom: 32, width: '100%', maxWidth: 500 }}>
          <SensorChart branchData={features[0]} title="Accelerometer (earth-fixed)" />
        </div>
      )}

      {/* MPI status card */}
      <div style={{
        background: '#0D1525', border: `2px solid ${cardColor}`, borderRadius: 16,
        padding: '32px 48px', textAlign: 'center',
        boxShadow: `0 0 30px ${cardColor}33`,
      }}>
        <p className="mono" style={{ fontSize: 22, color: cardColor, marginBottom: 8 }}>{title}</p>
        <p style={{ fontSize: 14, color: '#6B7E9E' }}>{subtitle}</p>
      </div>
    </div>
  )
}

function UVPhase({ uv, onSkip }) {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 40, position: 'relative' }}>
      <button className="btn-ghost" onClick={onSkip} style={{ position: 'absolute', top: 20, right: 20, fontSize: 12, padding: '6px 16px' }}>
        Skip →
      </button>

      <p className="mono text-cyan" style={{ fontSize: 16, marginBottom: 24 }}>Identity Verification</p>

      <ScoreDial score={uv.score} threshold={uv.threshold} />

      <p className="mono" style={{ fontSize: 14, color: '#E8EDF5', marginTop: 16 }}>
        Identity Score: <span className="text-cyan">{uv.score.toFixed(3)}</span>
      </p>
      <p className="mono" style={{ fontSize: 12, color: '#6B7E9E', marginTop: 4 }}>
        Threshold: {uv.threshold.toFixed(2)}
      </p>
    </div>
  )
}

export default function DemoPage({ userId, onComplete }) {
  const [phase, setPhase] = useState(1)  // 1=loading, 2=mpi, 3=uv
  const [result, setResult] = useState(null)
  const timer1Ref = useRef(null)
  const timer2Ref = useRef(null)

  useEffect(() => {
    runDemo(userId)
      .then(res => {
        setResult(res.data)
        setPhase(2)
        timer1Ref.current = setTimeout(() => {
          if (res.data.mpi && !res.data.mpi.is_unlock && !(res.data.mpi.note && res.data.mpi.note.includes('bypassed'))) {
            onComplete(res.data)
            return
          }
          setPhase(3)
          timer2Ref.current = setTimeout(() => onComplete(res.data), 2500)
        }, 2500)
      })
      .catch(() => onComplete({ final_decision: 'REJECT', mpi: {}, uv: { score: 0, threshold: 0.5 }, user_id: userId, pipeline_short_circuited: false, error: true }))

    return () => {
      if (timer1Ref.current) clearTimeout(timer1Ref.current)
      if (timer2Ref.current) clearTimeout(timer2Ref.current)
    }
  }, [userId, onComplete])

  const handleSkip = () => {
    if (timer1Ref.current) clearTimeout(timer1Ref.current)
    if (timer2Ref.current) clearTimeout(timer2Ref.current)
    if (result) onComplete(result)
  }

  if (phase === 1 || !result) return <LoadingPhase />
  if (phase === 2) return <MPIPhase mpi={result.mpi} features={result.sample?.features} onSkip={handleSkip} />
  return <UVPhase uv={result.uv} onSkip={handleSkip} />
}
