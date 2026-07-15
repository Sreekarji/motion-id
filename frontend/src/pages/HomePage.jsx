import { useState, useEffect } from 'react'
import { getUsers } from '../api'

const PIPELINE_STEPS = [
  { label: 'Pick Up Phone', icon: '📱' },
  { label: 'MPI: Unlock Pattern?', icon: '🔍' },
  { label: 'UV: Identity Match?', icon: '🔐' },
  { label: 'GRANTED / DENIED', icon: '✅' },
]

export default function HomePage({ onStartDemo }) {
  const [users, setUsers] = useState([])
  const [selected, setSelected] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getUsers()
      .then(res => {
        setUsers(res.data.users)
        if (res.data.users.length > 0) setSelected(String(res.data.users[0]))
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '60px 20px' }}>
      {/* Header */}
      <h1 className="mono" style={{ fontSize: 48, letterSpacing: 4, color: '#00D4FF', marginBottom: 8 }}>
        MOTION ID
      </h1>
      <p className="text-muted" style={{ fontSize: 16, marginBottom: 48, textAlign: 'center' }}>
        IMU-Based Passive Biometric Authentication
      </p>

      {/* Pipeline diagram */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: 48, flexWrap: 'wrap', justifyContent: 'center' }}>
        {PIPELINE_STEPS.map((step, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
            <div className="card-accent" style={{ padding: '16px 20px', textAlign: 'center', minWidth: 140 }}>
              <div style={{ fontSize: 28, marginBottom: 6 }}>{step.icon}</div>
              <div className="mono" style={{ fontSize: 12, color: '#00D4FF' }}>{step.label}</div>
            </div>
            {i < PIPELINE_STEPS.length - 1 && (
              <svg width="40" height="20" style={{ flexShrink: 0 }}>
                <line x1="0" y1="10" x2="30" y2="10" stroke="#00D4FF" strokeWidth="2"
                  strokeDasharray="6 4" style={{ animation: 'dash 0.5s linear infinite' }} />
                <polygon points="28,5 36,10 28,15" fill="#00D4FF" />
              </svg>
            )}
          </div>
        ))}
      </div>

      {/* How it works */}
      <p className="text-muted" style={{ fontSize: 14, marginBottom: 40, textAlign: 'center', maxWidth: 500 }}>
        Your phone's motion sensors create a unique signature every time you pick it up.
      </p>

      {/* User selector */}
      <div style={{ marginBottom: 24, width: '100%', maxWidth: 320 }}>
        <label className="mono" style={{ fontSize: 12, color: '#6B7E9E', display: 'block', marginBottom: 8 }}>
          SELECT USER FOR DEMO
        </label>
        <select
          value={selected}
          onChange={e => setSelected(e.target.value)}
          disabled={loading || users.length === 0}
          style={{
            width: '100%', padding: '12px 16px', background: '#0D1525', color: '#E8EDF5',
            border: '1px solid #1E2D4A', borderRadius: 8, fontSize: 16,
            fontFamily: 'JetBrains Mono, monospace', outline: 'none',
          }}
        >
          {loading ? <option>Loading users...</option> :
           users.map(u => <option key={u} value={u}>User {u}</option>)}
        </select>
      </div>

      {/* Run button */}
      <button
        className="btn-primary mono"
        disabled={loading || !selected}
        onClick={() => onStartDemo(Number(selected))}
        style={{ fontSize: 18, padding: '16px 40px' }}
      >
        ▶ Run Authentication Demo
      </button>

      {/* Footer */}
      <p className="text-muted" style={{ fontSize: 11, marginTop: 60 }}>
        Motion ID — Research Demo · arXiv:2302.01751
      </p>
    </div>
  )
}
