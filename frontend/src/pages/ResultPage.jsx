export default function ResultPage({ result, onReset }) {
  const accepted = result.final_decision === 'ACCEPT'
  const color = accepted ? '#00FF88' : '#FF4444'
  const icon = accepted ? '✓' : '✗'

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 40 }}>
      {/* Animated icon */}
      <svg width="100" height="100" viewBox="0 0 100 100" style={{ marginBottom: 24 }}>
        {accepted ? (
          <circle cx="50" cy="50" r="45" fill="none" stroke={color} strokeWidth="3"
            strokeDasharray="283" strokeDashoffset="283" style={{ animation: 'drawCircle 0.8s ease forwards' }}>
            <animate attributeName="stroke-dashoffset" from="283" to="0" dur="0.8s" fill="freeze" />
          </circle>
        ) : (
          <>
            <line x1="25" y1="25" x2="75" y2="75" stroke={color} strokeWidth="3"
              strokeDasharray="70" strokeDashoffset="70">
              <animate attributeName="stroke-dashoffset" from="70" to="0" dur="0.5s" fill="freeze" />
            </line>
            <line x1="75" y1="25" x2="25" y2="75" stroke={color} strokeWidth="3"
              strokeDasharray="70" strokeDashoffset="70">
              <animate attributeName="stroke-dashoffset" from="70" to="0" dur="0.5s" begin="0.3s" fill="freeze" />
            </line>
          </>
        )}
        {accepted && (
          <polyline points="30,52 45,65 72,35" fill="none" stroke={color} strokeWidth="3"
            strokeLinecap="round" strokeLinejoin="round"
            strokeDasharray="60" strokeDashoffset="60">
            <animate attributeName="stroke-dashoffset" from="60" to="0" dur="0.5s" begin="0.6s" fill="freeze" />
          </polyline>
        )}
      </svg>

      {/* Title */}
      <h1 className="mono" style={{ fontSize: 36, color: accepted ? '#00D4FF' : '#FF4444', marginBottom: 8, letterSpacing: 3 }}>
        {accepted ? 'IDENTITY VERIFIED' : 'ACCESS DENIED'}
      </h1>
      <p style={{ fontSize: 14, color: '#6B7E9E', marginBottom: 32 }}>
        {accepted ? 'Authentication successful. Access granted.' :
         result.pipeline_short_circuited ? 'Rejected at MPI stage' : 'Rejected at UV stage (score below threshold)'}
      </p>

      {/* Metrics */}
      <div className="card" style={{ minWidth: 320, marginBottom: 32 }}>
        {[
          ['User ID', result.user_id],
          ['MPI Confidence', `${((result.mpi?.confidence || 0) * 100).toFixed(1)}%`],
          ['UV Score', (result.uv?.score || 0).toFixed(4)],
          ['Threshold', (result.uv?.threshold || 0.5).toFixed(4)],
          ['Decision', `${result.final_decision} ${accepted ? '✓' : '✗'}`],
        ].map(([label, val]) => (
          <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid #1E2D4A' }}>
            <span className="mono" style={{ fontSize: 13, color: '#6B7E9E' }}>{label}</span>
            <span className="mono" style={{ fontSize: 13, color: '#E8EDF5' }}>{val}</span>
          </div>
        ))}
      </div>

      <button className="btn-ghost mono" onClick={onReset} style={{ marginBottom: 40 }}>
        ↺ Run Again
      </button>

      <p className="text-muted" style={{ fontSize: 11 }}>Motion ID — Research Demo</p>
    </div>
  )
}
