import React from 'react'

const NavigationItem = ({ 
  icon: Icon, 
  label, 
  isActive, 
  onClick 
}) => {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 w-full text-left p-2 rounded transition-colors ${
        isActive 
          ? 'bg-brand-900/30 text-brand-300' 
          : 'text-gray-300 hover:text-brand-300 hover:bg-brand-900/20'
      }`}
    >
      <Icon size={16} className="text-brand-400"/>
      {label}
    </button>
  )
}

export default NavigationItem