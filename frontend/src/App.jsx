import React, { useState } from 'react'
import { BarChart3, LineChart, FileText, Sparkles, Menu, X } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import RealTimeDashboard from './pages/RealTimeDashboard'

export default function App(){
  const [open, setOpen] = useState(true)
  const [useRealTimeDashboard, setUseRealTimeDashboard] = useState(true)
  
  if (useRealTimeDashboard) {
    return <RealTimeDashboard />
  }
  
  return (
    <div className="min-h-screen bg-surface-900">
      <header className="sticky top-0 z-40 border-b border-white/10 glass">
        <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button aria-label="Toggle navigation" className="lg:hidden text-gray-300 hover:text-white" onClick={()=>setOpen(!open)}>
              {open ? <X size={20}/> : <Menu size={20}/>} 
            </button>
            <div className="h-9 w-9 rounded-lg gradient-brand flex items-center justify-center text-white shadow-glow">
              <Sparkles size={18} />
            </div>
            <div>
              <h1 className="text-lg font-semibold leading-tight text-white">FinDocGPT</h1>
              <p className="text-xs text-gray-400">AI insights for financial documents</p>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-3">
            <button className="hidden sm:inline-flex px-3 py-2 text-sm rounded-md border border-white/10 text-gray-200 hover:bg-white/5 focus-neon">Sign in</button>
            <button className="inline-flex px-3 py-2 text-sm rounded-md text-white focus-neon gradient-brand">Get started</button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-6 py-8">
        <div className="grid grid-cols-12 gap-6">
          <aside className={`col-span-12 lg:col-span-3 ${open? 'block':'hidden lg:block'}`}>
            <div className="sticky top-24 space-y-3">
              <div className="p-4 rounded-xl glass shadow-card">
                <h2 className="text-sm font-semibold text-gray-200 mb-3">Modules</h2>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-center gap-2 text-gray-300"><LineChart size={16} className="text-brand-400"/> Forecasting</li>
                  <li className="flex items-center gap-2 text-gray-300"><BarChart3 size={16} className="text-brand-400"/> Sentiment</li>
                  <li className="flex items-center gap-2 text-gray-300"><FileText size={16} className="text-brand-400"/> Q&A</li>
                </ul>
              </div>
              <div className="p-4 rounded-xl border border-brand-900/30 bg-gradient-to-br from-brand-900/20 to-pink-900/10">
                <h3 className="text-sm font-semibold mb-1 text-white">Tip</h3>
                <p className="text-xs text-gray-400">Try ticker symbols like AAPL, MSFT, TSLA for fast forecasts.</p>
              </div>
            </div>
          </aside>
          <section className="col-span-12 lg:col-span-9">
            <Dashboard />
          </section>
        </div>
      </main>
      <footer className="border-t border-white/10 mt-12 glass">
        <div className="mx-auto max-w-7xl px-6 py-6 text-xs text-gray-400 flex items-center justify-between">
          <span>© {new Date().getFullYear()} FinDocGPT. All rights reserved.</span>
          <div className="flex gap-4">
            <a href="#privacy" className="hover:text-gray-200">Privacy</a>
            <a href="#terms" className="hover:text-gray-200">Terms</a>
          </div>
        </div>
      </footer>
    </div>
  )
}
