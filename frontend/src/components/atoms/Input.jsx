import React from 'react'

const Input = ({ 
  label, 
  error, 
  className = '', 
  ...props 
}) => {
  const inputClasses = `w-full rounded-md border border-white/10 bg-white/5 text-gray-100 placeholder:text-gray-500 px-3 py-2 text-sm focus-neon ${
    error ? 'border-red-500' : ''
  } ${className}`
  
  return (
    <div className="space-y-1">
      {label && (
        <label className="block text-sm font-medium text-gray-200">
          {label}
        </label>
      )}
      <input className={inputClasses} {...props} />
      {error && (
        <p className="text-sm text-red-400">{error}</p>
      )}
    </div>
  )
}

export default Input