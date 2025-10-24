import React from 'react'

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  disabled = false, 
  loading = false,
  onClick,
  className = '',
  ...props 
}) => {
  const baseClasses = 'inline-flex items-center justify-center rounded-md font-medium transition-colors focus-neon disabled:opacity-50 disabled:cursor-not-allowed'
  
  const variants = {
    primary: 'gradient-brand text-white hover:opacity-90',
    secondary: 'bg-emerald-600 text-white hover:bg-emerald-700',
    outline: 'border border-white/10 text-gray-200 hover:bg-white/5',
    ghost: 'text-gray-300 hover:text-brand-300 hover:bg-brand-900/20'
  }
  
  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-3 py-2 text-sm',
    lg: 'px-4 py-2.5 text-base'
  }
  
  const classes = `${baseClasses} ${variants[variant]} ${sizes[size]} ${className}`
  
  return (
    <button
      className={classes}
      disabled={disabled || loading}
      onClick={onClick}
      {...props}
    >
      {loading && (
        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2" />
      )}
      {children}
    </button>
  )
}

export default Button