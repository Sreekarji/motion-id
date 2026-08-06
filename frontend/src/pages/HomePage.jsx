import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getUsers, runVerification } from '../api'

export default function HomePage() {
  const navigate = useNavigate()
  const [users, setUsers] = useState([])
  const [selected, setSelected] = useState('')
  const [loadingUsers, setLoadingUsers] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let alive = true
    getUsers()
      .then(res => {
        if (!alive) return
        const list = res.data.users ?? []
        setUsers(list)
        if (list.length > 0) setSelected(String(list[0]))
      })
      .catch(err => {
        if (!alive) return
        setError(`Could not load users: ${err.message}`)
      })
      .finally(() => { if (alive) setLoadingUsers(false) })
    return () => { alive = false }
  }, [])

  const handleVerify = async () => {
    setRunning(true)
    setError('')
    try {
      const result = await runVerification(Number(selected))
      navigate('/verify', { state: { result, ranAt: Date.now() } })
    } catch (err) {
      setError(`Verification failed: ${err.message || 'the backend did not respond.'}`)
      setRunning(false)
    }
  }

  const disabled = loadingUsers || running || !selected

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center',
      justifyContent: 'center', padding: '40px 20px',
    }}>
      <div className="card" style={{ width: '100%', maxWidth: 460, padding: 36 }}>
        <h1 className="mono" style={{
          fontSize: 24, color: 'var(--accent)', letterSpacing: 1,
          marginBottom: 8, lineHeight: 1.3,
        }}>
          MotionID — Biometric Verification
        </h1>
        <p className="text-muted" style={{ fontSize: 14, marginBottom: 32 }}>
          Select a user to run full pipeline verification
        </p>

        <label className="mono" style={{
          fontSize: 11, color: 'var(--muted)', letterSpacing: 1.2,
          display: 'block', marginBottom: 8, textTransform: 'uppercase',
        }}>
          Claimed Identity
        </label>
        <select
          className="select-clean"
          value={selected}
          onChange={e => setSelected(e.target.value)}
          disabled={loadingUsers || running || users.length === 0}
          style={{ marginBottom: 24 }}
        >
          {loadingUsers
            ? <option>Loading users…</option>
            : users.length === 0
              ? <option>No users available</option>
              : users.map(u => <option key={u} value={u}>User {u}</option>)}
        </select>

        <button
          className="btn-primary mono"
          disabled={disabled}
          onClick={handleVerify}
          style={{
            width: '100%', fontSize: 15, display: 'flex',
            alignItems: 'center', justifyContent: 'center', gap: 10,
          }}
        >
          {running
            ? <><span className="spinner" />Running verification…</>
            : `Verify User ${selected || '—'}`}
        </button>

        {error && (
          <p className="mono" style={{
            color: 'var(--red)', fontSize: 12, marginTop: 14,
            lineHeight: 1.5, wordBreak: 'break-word',
          }}>
            {error}
          </p>
        )}

        {running && (
          <p className="text-muted" style={{ fontSize: 11, marginTop: 14, textAlign: 'center' }}>
            Scoring the genuine attempt plus every impostor — this takes a few seconds.
          </p>
        )}
      </div>
    </div>
  )
}
