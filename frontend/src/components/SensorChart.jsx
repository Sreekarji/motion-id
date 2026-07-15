import { LineChart, Line, XAxis, YAxis, ResponsiveContainer } from 'recharts'

export default function SensorChart({ branchData, title, height = 120 }) {
  // branchData: (4, 50) — channels 0,1,2 are X,Y,Z
  const chartData = Array.from({ length: 50 }, (_, t) => ({
    t,
    x: branchData[0][t],
    y: branchData[1][t],
    z: branchData[2][t],
  }))

  return (
    <div>
      {title && <p className="mono" style={{ fontSize: 11, color: '#6B7E9E', marginBottom: 6 }}>{title}</p>}
      <div style={{ background: '#0D1117', borderRadius: 8, padding: '8px 4px' }}>
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
            <XAxis dataKey="t" hide />
            <YAxis hide domain={['auto', 'auto']} />
            <Line type="monotone" dataKey="x" stroke="#00D4FF" dot={false} strokeWidth={1.5} isAnimationActive={false} />
            <Line type="monotone" dataKey="y" stroke="#FF007F" dot={false} strokeWidth={1.5} isAnimationActive={false} />
            <Line type="monotone" dataKey="z" stroke="#FFD700" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
