import React, { useState, useCallback, useRef } from 'react'
import { Upload } from 'lucide-react'
import { Button } from '../atoms'

const FileUploadArea = ({ 
  onFilesSelected, 
  maxFiles = 5, 
  accept = '.pdf,.txt,.html,.json',
  maxSize = 10 * 1024 * 1024 
}) => {
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    const files = Array.from(e.dataTransfer.files)
    onFilesSelected(files)
  }, [onFilesSelected])

  const handleFileInput = useCallback((e) => {
    const files = Array.from(e.target.files)
    onFilesSelected(files)
  }, [onFilesSelected])

  return (
    <div
      className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
        dragActive
          ? 'border-blue-400 bg-blue-900/20'
          : 'border-gray-600 hover:border-gray-500'
      }`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={accept}
        onChange={handleFileInput}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
      />
      
      <div className="space-y-4">
        <div className="mx-auto w-12 h-12 rounded-full bg-gray-800 flex items-center justify-center">
          <Upload className="text-gray-400" size={24} />
        </div>
        
        <div>
          <h3 className="text-lg font-semibold text-white mb-2">
            Upload Financial Documents
          </h3>
          <p className="text-gray-400 mb-4">
            Drag and drop files here, or click to browse
          </p>
          <p className="text-sm text-gray-500">
            Supports PDF, TXT, HTML, JSON • Max {maxFiles} files • Up to 10MB each
          </p>
        </div>
        
        <Button
          onClick={() => fileInputRef.current?.click()}
          variant="primary"
        >
          Choose Files
        </Button>
      </div>
    </div>
  )
}

export default FileUploadArea