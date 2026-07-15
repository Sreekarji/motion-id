import axios from 'axios'

const BASE = "http://localhost:8000"

export const getUsers = () => axios.get(`${BASE}/users`)
export const runDemo = (userId) => axios.get(`${BASE}/predict/demo/${userId}`)
export const getUserSample = (userId) => axios.get(`${BASE}/users/${userId}/sample`)
