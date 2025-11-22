import React, { useState, useCallback } from 'react'
import { File, X, CheckCircle, AlertCircle, FileText, Download, Eye } from 'lucide-react'
import { FileUploadArea } from '../molecules'
import { Card, Button } from '../atoms'
import { formatFileSize, validateFile, generateFileId, getFileIcon } from '../../utils'
import { documentService } from '../../services'

const DocumentUploader = ({ onUploadComplete, onUploadProgress, maxFiles = 5 }) => {
  const [uploadQueue, setUploadQueue] = useState([])
  const [uploadedDocuments, setUploadedDocuments] = useState([])

  const handleFiles = useCallback((files) => {
    const validFiles = files.filter(file => {
      const validation = validateFile(file)
      if (!validation.isValid) {
        console.warn('Invalid file:', validation.errors)
        return false
      }
      return true
    })

    if (validFiles.length + uploadedDocuments.length > maxFiles) {
      alert(`Maximum ${maxFiles} files allowed`)
      return
    }

    const newUploads = validFiles.map(file => ({
      id: generateFileId(),
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      status: 'pending',
      progress: 0,
      error: null,
      uploadedAt: null,
      documentId: null
    }))

    setUploadQueue(prev => [...prev, ...newUploads])
    
    newUploads.forEach(upload => {
      uploadFile(upload)
    })
  }, [uploadedDocuments.length, maxFiles])

  const uploadFile = async (upload) => {
    try {
      setUploadQueue(prev => prev.map(u => 
        u.id === upload.id ? { ...u, status: 'uploading', progress: 0 } : u
      ))

      const response = await documentService.uploadDocument(upload.file)
      
      setUploadQueue(prev => prev.filter(u => u.id !== upload.id))
      
      const completedDoc = {
        ...upload,
        status: 'completed',
        progress: 100,
        uploadedAt: new Date().toISOString(),
        documentId: response.document_id,
        insights: response.insights || null
      }
      
      setUploadedDocuments(prev => [...prev, completedDoc])
      
      if (onUploadComplete) {
        onUploadComplete(completedDoc)
      }

    } catch (error) {
      console.error('Upload error:', error)
      setUploadQueue(prev => prev.map(u => 
        u.id === upload.id 
          ? { ...u, status: 'error', error: error.message }
          : u
      ))
    }
  }

  const removeFromQueue = (uploadId) => {
    setUploadQueue(prev => prev.filter(u => u.id !== uploadId))
  }

  const removeDocument = (docId) => {
    setUploadedDocuments(prev => prev.filter(d => d.id !== docId))
  }

  const retryUpload = (upload) => {
    setUploadQueue(prev => prev.map(u => 
      u.id === upload.id 
        ? { ...u, status: 'pending', progress: 0, error: null }
        : u
    ))
    uploadFile(upload)
  }

  const renderFileIcon = (type) => {
    const iconName = getFileIcon(type)
    const iconProps = { size: 20 }
    
    switch (iconName) {
      case 'FileText':
        return <FileText className="text-red-400" {...iconProps} />
      default:
        return <File className="text-gray-400" {...iconProps} />
    }
  }

  return (
    <div className="space-y-6">
      <FileUploadArea 
        onFilesSelected={handleFiles}
        maxFiles={maxFiles}
      />

      {/* Upload Queue */}
      {uploadQueue.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-300">Uploading...</h4>
          {uploadQueue.map(upload => (
            <Card key={upload.id} className="p-4">
              <div className="flex items-center gap-3">
                {renderFileIcon(upload.type)}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-sm font-medium text-white truncate">
                      {upload.name}
                    </p>
                    <div className="flex items-center gap-2">
                      {upload.status === 'uploading' && (
                        <span className="text-xs text-blue-400">
                          {upload.progress}%
                        </span>
                      )}
                      {upload.status === 'error' && (
                        <Button
                          onClick={() => retryUpload(upload)}
                          variant="ghost"
                          size="sm"
                        >
                          Retry
                        </Button>
                      )}
                      <button
                        onClick={() => removeFromQueue(upload.id)}
                        className="p-1 text-gray-400 hover:text-red-400"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 text-xs text-gray-400">
                    <span>{formatFileSize(upload.size)}</span>
                    {upload.status === 'error' && (
                      <>
                        <span>•</span>
                        <span className="text-red-400">{upload.error}</span>
                      </>
                    )}
                  </div>
                  
                  {upload.status === 'uploading' && (
                    <div className="mt-2 w-full bg-gray-700 rounded-full h-1">
                      <div
                        className="bg-blue-500 h-1 rounded-full transition-all duration-300"
                        style={{ width: `${upload.progress}%` }}
                      />
                    </div>
                  )}
                </div>
                
                <div className="flex items-center">
                  {upload.status === 'uploading' && (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500" />
                  )}
                  {upload.status === 'error' && (
                    <AlertCircle className="text-red-400" size={16} />
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Uploaded Documents */}
      {uploadedDocuments.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-gray-300">
              Uploaded Documents ({uploadedDocuments.length})
            </h4>
            <span className="text-xs text-gray-500">
              {uploadedDocuments.length}/{maxFiles} files
            </span>
          </div>
          
          {uploadedDocuments.map(doc => (
            <Card key={doc.id} className="p-4 hover:border-white/20 transition-colors">
              <div className="flex items-center gap-3">
                {renderFileIcon(doc.type)}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-sm font-medium text-white truncate">
                      {doc.name}
                    </p>
                    <div className="flex items-center gap-2">
                      <CheckCircle className="text-green-400" size={16} />
                      <button
                        onClick={() => removeDocument(doc.id)}
                        className="p-1 text-gray-400 hover:text-red-400"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 text-xs text-gray-400">
                    <span>{formatFileSize(doc.size)}</span>
                    <span>•</span>
                    <span>Uploaded {new Date(doc.uploadedAt).toLocaleString()}</span>
                    {doc.documentId && (
                      <>
                        <span>•</span>
                        <span className="text-green-400">ID: {doc.documentId}</span>
                      </>
                    )}
                  </div>
                  
                  {doc.insights && (
                    <div className="mt-2 p-2 rounded bg-blue-900/20 border border-blue-900/30">
                      <p className="text-xs text-blue-200">
                        {doc.insights.summary || 'Document processed successfully'}
                      </p>
                    </div>
                  )}
                </div>
                
                <div className="flex items-center gap-1">
                  <button
                    className="p-2 text-gray-400 hover:text-blue-400 transition-colors"
                    title="View document"
                  >
                    <Eye size={16} />
                  </button>
                  <button
                    className="p-2 text-gray-400 hover:text-green-400 transition-colors"
                    title="Download"
                  >
                    <Download size={16} />
                  </button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}

export default DocumentUploader