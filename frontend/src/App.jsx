import React from 'react'
import Dashboard from './pages/Dashboard'

export default function App(){
  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-6xl mx-auto p-6">
        <h1 className="text-2xl font-bold mb-4">FinDocGPT — Hackathon Demo</h1>
        <Dashboard />
      </div>
    </div>
  )
}
