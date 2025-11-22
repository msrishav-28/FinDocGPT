import API from './api'

export const forecastService = {
  async getForecast(ticker) {
    const response = await API.get('/forecast', { params: { ticker } })
    return response.data
  },

  async getRecommendation(ticker, docId) {
    const formData = new FormData()
    formData.append('ticker', ticker)
    formData.append('doc_id', docId)
    
    const response = await API.post('/recommend', formData)
    return response.data
  },

  async getMarketData(ticker) {
    const response = await API.get('/market-data', { params: { ticker } })
    return response.data
  }
}