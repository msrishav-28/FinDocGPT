import React, { useState, useEffect } from 'react'
import { FileText, MessageSquare, GitCompare, Upload, Search, Filter, Grid, List } from 'lucide-react'
import DocumentUploader from './DocumentUploader'
import InteractiveQA from './InteractiveQA'
import DocumentComparison from './DocumentComparison'

const DocumentAnalysisInterface = ({ userId = 'demo-user' }) => {
  const [activeTab, setActiveTab] = useState('upload')
  const [documents, setDocuments] = useState([])
  const [selectedDocument, setSelectedDocument] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState('all')
  const [viewMode, setViewMode] = useState('grid')
  const [loading, setLoading] = useState(false)

  // Load existing documents on component mount
  useEffect(() => {
    loadDocuments()
  }, [])

  const loadDocuments = async () => {
    setLoading(true)
    try {
      const response = await fetch(`http://localhost:8000/api/documents/list?user_id=${userId}`)
      if (response.ok) {
        const data = await response.json()
        setDocuments(data.documents || [])
      }
    } catch (error) {
      console.error('Error loading documents:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleUploadComplete = (uploadedDoc) => {
    setDocuments(prev => [uploadedDoc, ...prev])
    
    // Auto-switch to Q&A tab and select the uploaded document
    if (uploadedDoc.documentId) {
      setSelectedDocument(uploadedDoc)
      setActiveTab('qa')
    }
  }

  const handleDocumentSelect = (document) => {
    setSelectedDocument(document)
    setActiveTab('qa')
  }

  const handleDeleteDocument = async (documentId) => {
    try {
      const response = await fetch(`http://localhost:8000/api/documents/${documentId}`, {
        method: 'DELETE'
      })
      
      if (response.ok) {
        setDocuments(prev => prev.filter(doc => doc.documentId !== documentId))
        if (selectedDocument?.documentId === documentId) {
          setSelectedDocument(null)
        }
      }
    } catch (error) {
      console.error('Error deleting document:', error)
    }
  }

  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.name.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesFilter = filterType === 'all' || doc.type === filterType
    return matchesSearch && matchesFilter
  })

  const tabs = [
    { id: 'upload', label: 'Upload', icon: Upload, description: 'Upload new documents' },
    { id: 'qa', label: 'Q&A', icon: MessageSquare, description: 'Ask questions about documents' },
    { id: 'compare', label: 'Compare', icon: GitCompare, description: 'Compare multiple documents' },
    { id: 'library', label: 'Library', icon: FileText, description: 'Browse document library' }
  ]

  const renderDocumentLibrary = () => (
    <div className="space-y-6">
      {/* Search and Filter Controls */}
      <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
        <div className="flex items-center gap-4 flex-1">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search documents..."
              className="w-full pl-10 pr-4 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
          </div>
          
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
          >
            <option value="all">All Types</option>
            <option value="application/pdf">PDF</option>
            <option value="text/plain">Text</option>
            <option value="text/html">HTML</option>
            <option value="application/json">JSON</option>
          </select>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode('grid')}
            className={`p-2 rounded-lg transition-colors ${
              viewMode === 'grid' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            <Grid size={16} />
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={`p-2 rounded-lg transition-colors ${
              viewMode === 'list' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            <List size={16} />
          </button>
        </div>
      </div>

      {/* Documents Display */}
      {loading ? (
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mr-3" />
          <span className="text-gray-400">Loading documents...</span>
        </div>
      ) : filteredDocuments.length === 0 ? (
        <div className="text-center p-8 rounded-xl glass">
          <FileText className="mx-auto text-gray-400 mb-4" size={48} />
          <p className="text-gray-400 mb-2">
            {searchQuery || filterType !== 'all' ? 'No documents match your search' : 'No documents uploaded yet'}
          </p>
          <p className="text-sm text-gray-500">
            {searchQuery || filterType !== 'all' ? 'Try adjusting your search or filter' : 'Upload some documents to get started'}
          </p>
        </div>
      ) : viewMode === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredDocuments.map(document => (
            <DocumentCard
              key={document.id}
              document={document}
              onSelect={handleDocumentSelect}
              onDelete={handleDeleteDocument}
            />
          ))}
        </div>
      ) : (
        <div className="space-y-2">
          {filteredDocuments.map(document => (
            <DocumentListItem
              key={document.id}
              document={document}
              onSelect={handleDocumentSelect}
              onDelete={handleDeleteDocument}
            />
          ))}
        </div>
      )}
    </div>
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-6 rounded-xl glass">
        <h2 className="text-2xl font-bold text-white mb-2">Document Analysis Center</h2>
        <p className="text-gray-400">
          Upload, analyze, and compare financial documents with AI-powered insights
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="flex flex-wrap gap-2">
        {tabs.map(tab => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-3 rounded-lg transition-colors ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <Icon size={18} />
              <div className="text-left">
                <div className="font-medium">{tab.label}</div>
                <div className="text-xs opacity-75">{tab.description}</div>
              </div>
            </button>
          )
        })}
      </div>

      {/* Tab Content */}
      <div className="min-h-[600px]">
        {activeTab === 'upload' && (
          <DocumentUploader
            onUploadComplete={handleUploadComplete}
            maxFiles={10}
          />
        )}

        {activeTab === 'qa' && (
          <div className="space-y-6">
            {selectedDocument ? (
              <div className="space-y-4">
                <div className="p-4 rounded-xl glass">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold text-white">{selectedDocument.name}</h3>
                      <p className="text-sm text-gray-400">
                        Uploaded {new Date(selectedDocument.uploadedAt).toLocaleString()}
                      </p>
                    </div>
                    <button
                      onClick={() => setSelectedDocument(null)}
                      className="px-3 py-2 text-sm bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600"
                    >
                      Change Document
                    </button>
                  </div>
                </div>
                
                <InteractiveQA
                  documentId={selectedDocument.documentId}
                  documentTitle={selectedDocument.name}
                  onQuestionSubmit={(question, response) => {
                    console.log('Q&A:', { question, response })
                  }}
                />
              </div>
            ) : (
              <div className="text-center p-8 rounded-xl glass">
                <MessageSquare className="mx-auto text-gray-400 mb-4" size={48} />
                <p className="text-gray-400 mb-4">Select a document to start asking questions</p>
                {documents.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-4xl mx-auto">
                    {documents.slice(0, 6).map(document => (
                      <button
                        key={document.id}
                        onClick={() => handleDocumentSelect(document)}
                        className="p-4 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors text-left"
                      >
                        <FileText className="text-blue-400 mb-2" size={20} />
                        <p className="font-medium text-white truncate">{document.name}</p>
                        <p className="text-sm text-gray-400">
                          {new Date(document.uploadedAt).toLocaleDateString()}
                        </p>
                      </button>
                    ))}
                  </div>
                ) : (
                  <button
                    onClick={() => setActiveTab('upload')}
                    className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Upload Your First Document
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'compare' && (
          <DocumentComparison documents={documents} />
        )}

        {activeTab === 'library' && renderDocumentLibrary()}
      </div>
    </div>
  )
}

// Document Card Component for Grid View
const DocumentCard = ({ document, onSelect, onDelete }) => {
  const getFileIcon = (type) => {
    switch (type) {
      case 'application/pdf':
        return <FileText className="text-red-400" size={24} />
      case 'text/plain':
        return <FileText className="text-gray-400" size={24} />
      case 'text/html':
        return <FileText className="text-blue-400" size={24} />
      case 'application/json':
        return <FileText className="text-green-400" size={24} />
      default:
        return <FileText className="text-gray-400" size={24} />
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="p-4 rounded-xl glass border border-white/10 hover:border-white/20 transition-colors">
      <div className="flex items-start gap-3">
        {getFileIcon(document.type)}
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-white truncate mb-1">{document.name}</h4>
          <p className="text-sm text-gray-400 mb-2">
            {formatFileSize(document.size)} • {new Date(document.uploadedAt).toLocaleDateString()}
          </p>
          
          {document.insights && (
            <p className="text-xs text-blue-300 mb-3 line-clamp-2">
              {document.insights.summary}
            </p>
          )}
          
          <div className="flex gap-2">
            <button
              onClick={() => onSelect(document)}
              className="flex-1 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Analyze
            </button>
            <button
              onClick={() => onDelete(document.documentId)}
              className="px-3 py-2 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              Delete
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// Document List Item Component for List View
const DocumentListItem = ({ document, onSelect, onDelete }) => {
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="p-4 rounded-xl glass border border-white/10 hover:border-white/20 transition-colors">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <FileText className="text-blue-400 flex-shrink-0" size={20} />
          <div className="flex-1 min-w-0">
            <h4 className="font-medium text-white truncate">{document.name}</h4>
            <div className="flex items-center gap-4 text-sm text-gray-400">
              <span>{formatFileSize(document.size)}</span>
              <span>{new Date(document.uploadedAt).toLocaleDateString()}</span>
              {document.insights && (
                <span className="text-blue-300">Analyzed</span>
              )}
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => onSelect(document)}
            className="px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Analyze
          </button>
          <button
            onClick={() => onDelete(document.documentId)}
            className="px-3 py-2 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  )
}

export default DocumentAnalysisInterface