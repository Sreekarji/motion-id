import axios from 'axios'

// Empty BASE = same-origin. FastAPI serves both the API and this frontend at :8000.
// Works for localhost AND for any ngrok/external URL without any code change.
const BASE = import.meta.env.VITE_API_URL ?? ""

const api = axios.create({ baseURL: BASE, timeout: 30000 })

export const getUsers = () => api.get(`/users`)

export async function runVerification(userId) {
  const res = await fetch(`/predict/demo/${userId}`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
