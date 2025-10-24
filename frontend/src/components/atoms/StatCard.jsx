import React from 'react'

const StatCard = ({ label, value, hint, tone = 'brand' }) => {
  const colors = {
    brand: 'from-brand-50 to-white border-brand-100',
    green: 'from-emerald-50 to-white border-emerald-100',
    gray: 'from-gray-50 to-white border-gray-100',
  }[tone] || 'from-gray-50 to-white border-gray-100'
  
  return (
    <div className={`rounded-xl border bg-gradient-to-br ${colors} p-4`}>
      <div className="text-xs text-gray-600">{label}</div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
      {hint && <div className="text-xs text-gray-500 mt-1">{hint}</div>}
    </div>
  )
}

export default StatCard