import axios from 'axios'

const baseURL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000') + '/api'
const API = axios.create({ baseURL, timeout: 30000 })

API.interceptors.response.use(r=>r, err=>{
	// Normalize network errors
	if(err.code === 'ECONNABORTED') {
		err.message = 'Request timed out. Please retry.'
	}
	return Promise.reject(err)
})

export default API