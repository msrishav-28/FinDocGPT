import React from 'react'
import { Menu, X, Sparkles } from 'lucide-react'
import { Button } from '../atoms'

const Header = ({ sidebarOpen, onToggleSidebar }) => {
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 glass">
      <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button 
            aria-label="Toggle navigation" 
            className="lg:hidden text-gray-300 hover:text-white" 
            onClick={onToggleSidebar}
          >
            {sidebarOpen ? <X size={20}/> : <Menu size={20}/>} 
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
          <Button variant="outline" size="sm">
            Sign in
          </Button>
          <Button variant="primary" size="sm">
            Get started
          </Button>
        </div>
      </div>
    </header>
  )
}

export default Header