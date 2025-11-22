import React, { useState, useEffect, useCallback } from 'react'
import { Bell, Settings, BarChart3, Activity } from 'lucide-react'
import { CustomizableDashboard, RealTimeMarketData } from '../components/organisms'
import { AlertManager, NotificationSystem } from '../components/molecules'
import { useWebSocket } from '../hooks'

const RealTimeDashboard = () => {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [notifications, setNotifications] = useState([])
  const [unreadCount, setUnreadCount] = useState(0)
  const userId = 'demo-user'

  // WebSocket connection for real-time notifications
  const wsUrl = `ws://localhost:8000/api/ws/connect?user_id=${userId}`
  
  const handleWebSocketMessage = useCallback((message) => {
    if (message.type === 'alert') {
      const notification = {
        id: message.data.id || `alert_${Date.now()}`,
        title: message.data.title,
        message: message.data.message,
        severity: message.data.severity,
        type: message.data.type,
        timestamp: message.data.timestamp || new Date().toISOString(),
        data: message.data.data || {},
        autoDismiss: message.data.severity !== 'critical' // Critical alerts require manual dismissal
      }
      
      setNotifications(prev => [notification, ...prev])
      setUnreadCount(prev => prev + 1)
    }
  }, [])

  const { connectionStatus, sendMessage } = useWebSocket(wsUrl, {
    onMessage: handleWebSocketMessage,
    maxReconnectAttempts: 5,
    reconnectInterval: 3000
  })

  const handleNotificationDismiss = useCallback((notificationId) => {
    setNotifications(prev => prev.filter(n => n.id !== notificationId))
    setUnreadCount(prev => Math.max(0, prev - 1))
  }, [])

  const clearAllNotifications = useCallback(() => {
    setNotifications([])
    setUnreadCount(0)
  }, [])

  // Test notification function for demo purposes
  const testNotification = useCallback(() => {
    const testAlert = {
      id: `test_${Date.now()}`,
      title: 'Test Alert',
      message: 'This is a test notification to demonstrate the alert system.',
      severity: 'medium',
      type: 'system_alert',
      timestamp: new Date().toISOString(),
      data: { ticker: 'AAPL', current_price: 150.25, price_change_percent: 2.5 }
    }
    
    setNotifications(prev => [testAlert, ...prev])
    setUnreadCount(prev => prev + 1)
  }, [])

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'market', label: 'Market Data', icon: Activity },
    { id: 'alerts', label: 'Alerts', icon: Bell }
  ]

  return (
    <div className="min-h-screen bg-surface-900">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-white/10 glass">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <h1 className="text-xl font-semibold text-white">
                Financial Intelligence Platform
              </h1>
              
              {/* Connection Status */}
              <div className="flex items-center gap-2 text-sm">
                <div className={`w-2 h-2 rounded-full ${
                  connectionStatus === 'Connected' ? 'bg-green-400' : 'bg-red-400'
                }`}></div>
                <span className="text-gray-400">{connectionStatus}</span>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Notification Bell */}
              <div className="relative">
                <button
                  onClick={() => setActiveTab('alerts')}
                  className={`p-2 rounded-lg transition-colors ${
                    unreadCount > 0 
                      ? 'text-orange-400 bg-orange-900/20' 
                      : 'text-gray-400 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <Bell size={20} />
                  {unreadCount > 0 && (
                    <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                      {unreadCount > 9 ? '9+' : unreadCount}
                    </span>
                  )}
                </button>
              </div>
              
              {/* Test Notification Button (for demo) */}
              <button
                onClick={testNotification}
                className="px-3 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors text-sm"
              >
                Test Alert
              </button>
              
              {/* Clear Notifications */}
              {notifications.length > 0 && (
                <button
                  onClick={clearAllNotifications}
                  className="px-3 py-2 rounded-md bg-gray-600 text-white hover:bg-gray-700 transition-colors text-sm"
                >
                  Clear All
                </button>
              )}
            </div>
          </div>
          
          {/* Navigation Tabs */}
          <div className="flex gap-1 mt-4">
            {tabs.map(tab => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                    activeTab === tab.id
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <Icon size={16} />
                  {tab.label}
                  {tab.id === 'alerts' && unreadCount > 0 && (
                    <span className="bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                      {unreadCount > 9 ? '9+' : unreadCount}
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-6 py-8">
        {activeTab === 'dashboard' && (
          <CustomizableDashboard />
        )}
        
        {activeTab === 'market' && (
          <RealTimeMarketData 
            watchlist={['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA']}
            userId={userId}
          />
        )}
        
        {activeTab === 'alerts' && (
          <AlertManager userId={userId} />
        )}
      </main>

      {/* Notification System */}
      <NotificationSystem
        notifications={notifications}
        onDismiss={handleNotificationDismiss}
        maxNotifications={5}
      />
    </div>
  )
}

export default RealTimeDashboard