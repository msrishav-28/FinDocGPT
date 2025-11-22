import API from './api'

export const documentService = {
  async uploadDocument(file, metadata = {}) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('metadata', JSON.stringify({
      filename: file.name,
      fileType: file.type,
      uploadedAt: new Date().toISOString(),
      ...metadata
    }))

    const response = await API.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  async getDocuments() {
    const response = await API.get('/documents')
    return response.data
  },

  async getDocument(documentId) {
    const response = await API.get(`/documents/${documentId}`)
    return response.data
  },

  async deleteDocument(documentId) {
    const response = await API.delete(`/documents/${documentId}`)
    return response.data
  },

  async analyzeDocument(documentId) {
    const response = await API.post(`/documents/${documentId}/analyze`)
    return response.data
  }
}