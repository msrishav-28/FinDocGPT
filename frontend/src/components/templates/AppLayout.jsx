import React, { useState } from 'react'
import { Header, Footer, Navigation } from '../organisms'

const AppLayout = ({ children, activeView, onViewChange }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="min-h-screen bg-surface-900">
      <Header sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />
      <main className="mx-auto max-w-7xl px-6 py-8">
        <div className="grid grid-cols-12 gap-6">
          <aside className={`col-span-12 lg:col-span-3 ${sidebarOpen ? 'block' : 'hidden lg:block'}`}>
            <Navigation activeView={activeView} onViewChange={onViewChange} />
          </aside>
          <section className="col-span-12 lg:col-span-9">
            {children}
          </section>
        </div>
      </main>
      <Footer />
    </div>
  )
}

export default AppLayout