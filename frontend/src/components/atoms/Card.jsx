import React from 'react'

const Card = ({ 
  children, 
  className = '', 
  hover = false,
  ...props 
}) => {
  const baseClasses = 'rounded-xl glass shadow-card animate-card-in'
  const hoverClasses = hover ? 'hover-lift' : ''
  const classes = `${baseClasses} ${hoverClasses} ${className}`
  
  return (
    <div className={classes} {...props}>
      {children}
    </div>
  )
}

export default Card