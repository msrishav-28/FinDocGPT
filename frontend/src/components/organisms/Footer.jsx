import React from 'react'

const Footer = () => {
  return (
    <footer className="border-t border-white/10 mt-12 glass">
      <div className="mx-auto max-w-7xl px-6 py-6 text-xs text-gray-400 flex items-center justify-between">
        <span>© {new Date().getFullYear()} FinDocGPT. All rights reserved.</span>
        <div className="flex gap-4">
          <a href="#privacy" className="hover:text-gray-200">Privacy</a>
          <a href="#terms" className="hover:text-gray-200">Terms</a>
        </div>
      </div>
    </footer>
  )
}

export default Footer