import React from 'react'

const LoadingSpinner = ({ size = 'md', className = '' }) => {
  const sizes = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8'
  }
  
  return (
    <div className={`animate-spin rounded-full border-b-2 border-current ${sizes[size]} ${className}`} />
  )
}

export default LoadingSpinner