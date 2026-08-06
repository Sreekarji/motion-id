import { useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import ScoreDial from '../components/ScoreDial'

const GREEN = '#00FF88'
const RED = '#FF4444'

function fmtTime(ts) {
  return new Date(ts).toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

// UV scores are cosine similarities in [-1, 1]; the meter is a 0-100% track.
const toPct = (v) => Math.max(0, Math.min(100, ((v + 1) / 2) * 100))

function BackButton({ onClick, style }) {
  return (
    <button className="btn-ghost mono" onClick={onClick}
      style={{ fontSize: 13, whiteSpace: 'nowrap', ...style }}>
      ← Verify Another User
    </button>
  )
}

function ImpostorCard({ imp, index }) {
  const failed = imp.decision === 'ACCEPT'
  const color = failed ? RED : GREEN
  return (
    <div
      className={`impostor-card fade-up${failed ? ' is-failure' : ''}`}
      style={{ animationDelay: `${Math.min(index * 45, 700)}ms` }}
    >
      <div style={{
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', marginBottom: 14, gap: 8,
      }}>
        <span className="mono" style={{ fontSize: 14, color: 'var(--text)' }}>
          User {imp.impostor_user_id}
        </span>
        <span className={`badge ${failed ? 'badge-fail' : 'badge-ok'}`}>
          {imp.decision}
        </span>
      </div>

      <div className="meter">
        <div className="meter-fill"
          style={{ width: `${toPct(imp.uv_score)}%`, background: color }} />
        <div className="meter-mark"
          style={{ left: `calc(${toPct(imp.threshold)}% - 1px)` }}
          title={`threshold ${imp.threshold.toFixed(4)}`} />
      </div>

      <div className="mono" style={{
        display: 'flex', justifyContent: 'space-between',
        fontSize: 11, color: 'var(--muted)', marginTop: 8,
      }}>
        <span>score <span style={{ color }}>{imp.uv_score.toFixed(4)}</span></span>
        <span>thr {imp.threshold.toFixed(3)}</span>
      </div>
    </div>
  )
}

export default function VerificationPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const result = location.state?.result
  const ranAt = location.state?.ranAt

  // Direct navigation to /verify has no router state to render.
  useEffect(() => {
    if (!result) navigate('/', { replace: true })
  }, [result, navigate])

  if (!result) return null

  const { claimed_user_id: uid, genuine, impostors = [], summary } = result
  const goHome = () => navigate('/')

  const accepted = genuine.decision === 'ACCEPT'
  const genColor = accepted ? GREEN : RED
  const allRejected = summary.incorrectly_accepted === 0 && summary.total_impostors > 0

  return (
    <div style={{ minHeight: '100vh', padding: '28px 32px 56px' }}>
      <div style={{ maxWidth: 1180, margin: '0 auto' }}>

        {/* ── Header ─────────────────────────────────────────────── */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          gap: 16, paddingBottom: 18, marginBottom: 28,
          borderBottom: '1px solid var(--border)',
        }}>
          <div style={{ flex: '1 1 0', display: 'flex' }}>
            <BackButton onClick={goHome} />
          </div>
          <h1 className="mono" style={{
            fontSize: 18, color: 'var(--text)', letterSpacing: 0.5,
            textAlign: 'center', whiteSpace: 'nowrap',
          }}>
            User {uid} — Full Verification
          </h1>
          <div style={{ flex: '1 1 0', textAlign: 'right' }}>
            <span className="mono text-muted" style={{ fontSize: 11 }}>
              {fmtTime(ranAt ?? Date.now())}
            </span>
          </div>
        </div>

        {/* ── Genuine attempt ────────────────────────────────────── */}
        <div className="section-label">
          Genuine Attempt — User {uid}'s own motion data
        </div>

        <div style={{
          background: 'var(--panel)',
          border: `1px solid ${genColor}`,
          boxShadow: `0 0 28px ${accepted ? 'rgba(0,255,136,0.10)' : 'rgba(255,68,68,0.10)'}`,
          borderRadius: 14, padding: '26px 32px', marginBottom: 32,
          display: 'flex', alignItems: 'center', gap: 40, flexWrap: 'wrap',
        }}>
          <div style={{ flex: '0 0 auto' }}>
            <ScoreDial score={genuine.uv_score} threshold={genuine.threshold} />
          </div>

          <div style={{ flex: '1 1 260px', minWidth: 240 }}>
            <div className="mono" style={{
              fontSize: 34, fontWeight: 700, color: genColor,
              letterSpacing: 2, marginBottom: 10,
            }}>
              {genuine.decision}
            </div>
            <p className="text-muted" style={{ fontSize: 13, marginBottom: 18, lineHeight: 1.5 }}>
              {accepted
                ? "The user's own motion sample was correctly matched to their enrolled template."
                : 'The genuine sample scored below threshold — this is a false rejection.'}
            </p>
            <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
              {[
                ['UV Score', genuine.uv_score.toFixed(4), genColor],
                ['Threshold', genuine.threshold.toFixed(4), 'var(--amber)'],
                ['Trial', `${genuine.trial_index} of ${genuine.n_trials_total}`, 'var(--text)'],
              ].map(([label, value, color]) => (
                <div key={label}>
                  <div className="mono" style={{ fontSize: 17, color }}>{value}</div>
                  <div className="stat-label">{label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── Summary ────────────────────────────────────────────── */}
        <div style={{ display: 'flex', gap: 14, marginBottom: 32, flexWrap: 'wrap' }}>
          {[
            ['Impostors Tested', summary.total_impostors, 'var(--text)'],
            ['Correctly Rejected', summary.correctly_rejected, GREEN],
            ['Incorrectly Accepted', summary.incorrectly_accepted,
              summary.incorrectly_accepted === 0 ? GREEN : RED],
            ['FAR', summary.far_display, 'var(--accent)'],
          ].map(([label, value, color]) => (
            <div key={label} className="stat-card">
              <div className="stat-value" style={{ color }}>{value}</div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>

        {/* ── Impostors ──────────────────────────────────────────── */}
        <div className="section-label">
          Impostor Attempts — Other users run through User {uid}'s model
        </div>

        {allRejected && (
          <div className="banner-ok">
            <span>✓</span>
            <span>All impostors correctly rejected</span>
          </div>
        )}

        {impostors.length === 0 ? (
          <p className="text-muted" style={{ fontSize: 13 }}>
            No other enrolled users were available to test as impostors.
          </p>
        ) : (
          <div className="impostor-grid">
            {impostors.map((imp, i) => (
              <ImpostorCard key={imp.impostor_user_id} imp={imp} index={i} />
            ))}
          </div>
        )}

        {/* ── Footer ─────────────────────────────────────────────── */}
        <div style={{ textAlign: 'center', marginTop: 48 }}>
          <BackButton onClick={goHome} />
        </div>
      </div>
    </div>
  )
}
