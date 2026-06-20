import axios from 'axios'

export const API_BASE = 'http://localhost:5000/api'

const api = axios.create({
  baseURL: API_BASE,
  // No client-side timeout — LLM + LangGraph calls can take 60-120s
  timeout: 0,
})

// ── Response interceptor — humanize errors ──────────────────────────────────
api.interceptors.response.use(
  res => res,
  err => {
    if (!err.response) {
      // No response at all → Flask not running
      err.friendlyMessage =
        '⚠️ Cannot reach the backend server.\n' +
        'Make sure Flask is running:\n' +
        '  /home/zombie/Documents/Agentic/bin/python flask_app.py'
    } else if (err.response.status === 400) {
      err.friendlyMessage = err.response.data?.error || 'Bad request (400)'
    } else if (err.response.status === 500) {
      err.friendlyMessage = '❌ Server error (500) — check Flask logs.'
    } else {
      err.friendlyMessage = err.response.data?.error || err.message
    }
    return Promise.reject(err)
  }
)

export default api
