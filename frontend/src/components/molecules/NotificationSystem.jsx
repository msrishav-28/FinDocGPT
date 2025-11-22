import React, { useState, useEffect, useCallback } from 'react'
import { X, Bell, AlertTriangle, Info, CheckCircle, AlertCircle } from 'lucide-react'

const NotificationSystem = ({ notifications = [], onDismiss, maxNotifications = 5 }) => {
  const [visibleNotifications, setVisibleNotifications] = useState([])

  useEffect(() => {
    // Keep only the most recent notifications
    const recent = notifications.slice(-maxNotifications)
    setVisibleNotifications(recent)
  }, [notifications, maxNotifications])

  const handleDismiss = useCallback((notificationId) => {
    setVisibleNotifications(prev => 
      prev.filter(notification => notification.id !== notificationId)
    )
    if (onDismiss) {
      onDismiss(notificationId)
    }
  }, [onDismiss])

  const getNotificationIcon = (severity) => {
    switch (severity) {
      case 'critical':
        return <AlertTriangle className="text-red-400" size={20} />
      case 'high':
        return <AlertCircle className="text-orange-400" size={20} />
      case 'medium':
        return <Info className="text-blue-400" size={20} />
      case 'low':
        return <CheckCircle className="text-green-400" size={20} />
      default:
        return <Bell className="text-gray-400" size={20} />
    }
  }

  const getNotificationStyles = (severity) => {
    switch (severity) {
      case 'critical':
        return 'border-red-500 bg-red-900/20 shadow-red-500/20'
      case 'high':
        return 'border-orange-500 bg-orange-900/20 shadow-orange-500/20'
      case 'medium':
        return 'border-blue-500 bg-blue-900/20 shadow-blue-500/20'
      case 'low':
        return 'border-green-500 bg-green-900/20 shadow-green-500/20'
      default:
        return 'border-gray-500 bg-gray-900/20 shadow-gray-500/20'
    }
  }

  if (visibleNotifications.length === 0) {
    return null
  }

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm">
      {visibleNotifications.map((notification, index) => (
        <NotificationItem
          key={notification.id}
          notification={notification}
          onDismiss={handleDismiss}
          getIcon={getNotificationIcon}
          getStyles={getNotificationStyles}
          index={index}
        />
      ))}
    </div>
  )
}

const NotificationItem = ({ notification, onDismiss, getIcon, getStyles, index }) => {
  const [isVisible, setIsVisible] = useState(false)
  const [isExiting, setIsExiting] = useState(false)

  useEffect(() => {
    // Animate in
    const timer = setTimeout(() => setIsVisible(true), index * 100)
    return () => clearTimeout(timer)
  }, [index])

  useEffect(() => {
    // Auto-dismiss after delay
    if (notification.autoDismiss !== false) {
      const dismissDelay = notification.duration || 5000
      const timer = setTimeout(() => {
        handleDismiss()
      }, dismissDelay)
      return () => clearTimeout(timer)
    }
  }, [notification])

  const handleDismiss = useCallback(() => {
    setIsExiting(true)
    setTimeout(() => {
      onDismiss(notification.id)
    }, 300)
  }, [notification.id, onDismiss])

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div
      className={`
        transform transition-all duration-300 ease-out
        ${isVisible && !isExiting ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}
        ${isExiting ? 'scale-95' : 'scale-100'}
      `}
    >
      <div
        className={`
          p-4 rounded-lg border shadow-lg backdrop-blur-sm
          ${getStyles(notification.severity)}
          hover:shadow-xl transition-shadow duration-200
        `}
      >
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 mt-0.5">
            {getIcon(notification.severity)}
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between gap-2">
              <h4 className="text-sm font-semibold text-white truncate">
                {notification.title}
              </h4>
              <button
                onClick={handleDismiss}
                className="flex-shrink-0 text-gray-400 hover:text-white transition-colors"
              >
                <X size={16} />
              </button>
            </div>
            
            <p className="text-sm text-gray-300 mt-1 leading-relaxed">
              {notification.message}
            </p>
            
            <div className="flex items-center justify-between mt-2">
              <span className="text-xs text-gray-400">
                {formatTimestamp(notification.timestamp)}
              </span>
              
              {notification.type && (
                <span className="text-xs px-2 py-1 rounded-full bg-white/10 text-gray-300">
                  {notification.type.replace('_', ' ')}
                </span>
              )}
            </div>
            
            {notification.data && notification.data.ticker && (
              <div className="mt-2 p-2 rounded bg-white/5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">Ticker:</span>
                  <span className="text-white font-mono">{notification.data.ticker}</span>
                </div>
                {notification.data.current_price && (
                  <div className="flex items-center justify-between text-xs mt-1">
                    <span className="text-gray-400">Price:</span>
                    <span className="text-white">
                      ${notification.data.current_price.toFixed(2)}
                    </span>
                  </div>
                )}
                {notification.data.price_change_percent && (
                  <div className="flex items-center justify-between text-xs mt-1">
                    <span className="text-gray-400">Change:</span>
                    <span className={`${
                      notification.data.price_change_percent >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {notification.data.price_change_percent >= 0 ? '+' : ''}
                      {notification.data.price_change_percent.toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default NotificationSystem