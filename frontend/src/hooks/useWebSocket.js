import { useState, useEffect, useRef, useCallback } from 'react'

const useWebSocket = (url, options = {}) => {
  const [socket, setSocket] = useState(null)
  const [lastMessage, setLastMessage] = useState(null)
  const [readyState, setReadyState] = useState(0) // 0: CONNECTING, 1: OPEN, 2: CLOSING, 3: CLOSED
  const [connectionStatus, setConnectionStatus] = useState('Connecting')
  
  const reconnectTimeoutRef = useRef(null)
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = options.maxReconnectAttempts || 5
  const reconnectInterval = options.reconnectInterval || 3000
  
  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url)
      
      ws.onopen = () => {
        setReadyState(1)
        setConnectionStatus('Connected')
        reconnectAttemptsRef.current = 0
        console.log('WebSocket connected')
      }
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setLastMessage(data)
          if (options.onMessage) {
            options.onMessage(data)
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }
      
      ws.onclose = (event) => {
        setReadyState(3)
        setConnectionStatus('Disconnected')
        console.log('WebSocket disconnected:', event.code, event.reason)
        
        // Attempt to reconnect if not manually closed
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1
          setConnectionStatus(`Reconnecting... (${reconnectAttemptsRef.current}/${maxReconnectAttempts})`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setConnectionStatus('Connection failed')
        }
      }
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setConnectionStatus('Connection error')
      }
      
      setSocket(ws)
      
    } catch (error) {
      console.error('Error creating WebSocket connection:', error)
      setConnectionStatus('Connection failed')
    }
  }, [url, options, maxReconnectAttempts, reconnectInterval])
  
  useEffect(() => {
    connect()
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (socket) {
        socket.close(1000, 'Component unmounting')
      }
    }
  }, [connect])
  
  const sendMessage = useCallback((message) => {
    if (socket && readyState === 1) {
      const messageString = typeof message === 'string' ? message : JSON.stringify(message)
      socket.send(messageString)
      return true
    }
    console.warn('WebSocket is not connected. Message not sent:', message)
    return false
  }, [socket, readyState])
  
  const disconnect = useCallback(() => {
    if (socket) {
      socket.close(1000, 'Manual disconnect')
    }
  }, [socket])
  
  const reconnect = useCallback(() => {
    if (socket) {
      socket.close()
    }
    reconnectAttemptsRef.current = 0
    connect()
  }, [socket, connect])
  
  return {
    socket,
    lastMessage,
    readyState,
    connectionStatus,
    sendMessage,
    disconnect,
    reconnect
  }
}

export default useWebSocket