export const getFileIcon = (type) => {
  const iconMap = {
    'application/pdf': 'FileText',
    'text/plain': 'File',
    'text/html': 'File',
    'application/json': 'File'
  }
  return iconMap[type] || 'File'
}

export const validateFile = (file, options = {}) => {
  const {
    maxSize = 10 * 1024 * 1024, // 10MB default
    allowedTypes = ['application/pdf', 'text/plain', 'text/html', 'application/json']
  } = options

  const errors = []

  if (!allowedTypes.includes(file.type)) {
    errors.push(`Invalid file type: ${file.type}`)
  }

  if (file.size > maxSize) {
    errors.push(`File too large: ${file.size} bytes`)
  }

  return {
    isValid: errors.length === 0,
    errors
  }
}

export const generateFileId = () => {
  return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}