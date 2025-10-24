import React from 'react'
import { BarChart3, FileText, Target } from 'lucide-react'
import { NavigationItem } from '../molecules'
import { Card } from '../atoms'

const Navigation = ({ activeView, onViewChange }) => {
  const views = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'documents', label: 'Documents', icon: FileText },
    { id: 'recommendations', label: 'Recommendations', icon: Target }
  ]

  return (
    <div className="sticky top-24 space-y-3">
      <Card className="p-4">
        <h2 className="text-sm font-semibold text-gray-200 mb-3">Modules</h2>
        <ul className="space-y-2 text-sm">
          {views.map(view => (
            <li key={view.id}>
              <NavigationItem
                icon={view.icon}
                label={view.label}
                isActive={activeView === view.id}
                onClick={() => onViewChange(view.id)}
              />
            </li>
          ))}
        </ul>
      </Card>
      
      <div className="p-4 rounded-xl border border-brand-900/30 bg-gradient-to-br from-brand-900/20 to-pink-900/10">
        <h3 className="text-sm font-semibold mb-1 text-white">Tip</h3>
        <p className="text-xs text-gray-400">
          Try ticker symbols like AAPL, MSFT, TSLA for fast forecasts.
        </p>
      </div>
    </div>
  )
}

export default Navigation