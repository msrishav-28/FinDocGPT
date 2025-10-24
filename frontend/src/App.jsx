import React, { useState } from 'react'
import Dashboard from './pages/Dashboard'
import RealTimeDashboard from './pages/RealTimeDashboard'
import { DocumentAnalysisInterface, InvestmentRecommendationInterface, AppLayout } from './components'

export default function App(){
  const [activeView, setActiveView] = useState('dashboard') // dashboard, documents, recommendations
  
  const views = [
    { id: 'dashboard', label: 'Dashboard', component: RealTimeDashboard },
    { id: 'documents', label: 'Documents', component: DocumentAnalysisInterface },
    { id: 'recommendations', label: 'Recommendations', component: InvestmentRecommendationInterface }
  ]
  
  const currentView = views.find(v => v.id === activeView)
  const Component = currentView?.component || RealTimeDashboard
  
  return (
    <AppLayout activeView={activeView} onViewChange={setActiveView}>
      <Component />
    </AppLayout>
  )
}
