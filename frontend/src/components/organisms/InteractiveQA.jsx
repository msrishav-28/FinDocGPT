import React, { useState, useRef, useEffect } from 'react'
import { Send, MessageCircle, Bot, User, Lightbulb, Copy, ThumbsUp, ThumbsDown, RotateCcw } from 'lucide-react'

const InteractiveQA = ({ documentId, documentTitle, onQuestionSubmit }) => {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      type: 'system',
      content: `I'm ready to answer questions about "${documentTitle}". Ask me anything about the document's content, financial metrics, risks, opportunities, or management commentary.`,
      timestamp: new Date().toISOString(),
      suggestions: [
        "What are the key financial highlights?",
        "What risks are mentioned in the document?",
        "What is management's outlook?",
        "How did revenue perform this quarter?"
      ]
    }
  ])
  const [currentQuestion, setCurrentQuestion] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [suggestedQuestions, setSuggestedQuestions] = useState([
    "What are the main revenue drivers?",
    "What challenges does the company face?",
    "What are the growth opportunities?",
    "How is the company's financial health?",
    "What did management say about future prospects?",
    "Are there any regulatory concerns mentioned?"
  ])
  
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (question = currentQuestion) => {
    if (!question.trim() || isLoading) return

    const userMessage = {
      id: `user_${Date.now()}`,
      type: 'user',
      content: question.trim(),
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setCurrentQuestion('')
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/documents/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          document_id: documentId,
          question: question.trim()
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      const botMessage = {
        id: `bot_${Date.now()}`,
        type: 'bot',
        content: data.answer,
        timestamp: new Date().toISOString(),
        confidence: data.confidence,
        sources: data.sources || [],
        relatedQuestions: data.related_questions || [],
        metadata: data.metadata || {}
      }

      setMessages(prev => [...prev, botMessage])

      // Update suggested questions with related questions
      if (data.related_questions && data.related_questions.length > 0) {
        setSuggestedQuestions(prev => {
          const newSuggestions = [...data.related_questions, ...prev]
          return [...new Set(newSuggestions)].slice(0, 6) // Keep unique, max 6
        })
      }

      if (onQuestionSubmit) {
        onQuestionSubmit(question.trim(), data)
      }

    } catch (error) {
      console.error('Error asking question:', error)
      
      const errorMessage = {
        id: `error_${Date.now()}`,
        type: 'error',
        content: 'Sorry, I encountered an error while processing your question. Please try again.',
        timestamp: new Date().toISOString()
      }

      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
  }

  const rateResponse = (messageId, rating) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, userRating: rating }
        : msg
    ))
  }

  const clearConversation = () => {
    setMessages([
      {
        id: 'welcome',
        type: 'system',
        content: `I'm ready to answer questions about "${documentTitle}". Ask me anything about the document's content, financial metrics, risks, opportunities, or management commentary.`,
        timestamp: new Date().toISOString(),
        suggestions: [
          "What are the key financial highlights?",
          "What risks are mentioned in the document?",
          "What is management's outlook?",
          "How did revenue perform this quarter?"
        ]
      }
    ])
  }

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }

  return (
    <div className="flex flex-col h-full max-h-[600px] rounded-xl glass overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <div className="flex items-center gap-2">
          <MessageCircle className="text-blue-400" size={20} />
          <h3 className="font-semibold text-white">Document Q&A</h3>
        </div>
        <button
          onClick={clearConversation}
          className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
          title="Clear conversation"
        >
          <RotateCcw size={16} />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map(message => (
          <div key={message.id} className={`flex gap-3 ${
            message.type === 'user' ? 'justify-end' : 'justify-start'
          }`}>
            {message.type !== 'user' && (
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                message.type === 'bot' ? 'bg-blue-600' : 
                message.type === 'error' ? 'bg-red-600' : 'bg-gray-600'
              }`}>
                {message.type === 'bot' ? <Bot size={16} /> : <MessageCircle size={16} />}
              </div>
            )}
            
            <div className={`max-w-[80%] ${
              message.type === 'user' ? 'order-first' : ''
            }`}>
              <div className={`p-3 rounded-lg ${
                message.type === 'user' 
                  ? 'bg-blue-600 text-white ml-auto' 
                  : message.type === 'error'
                  ? 'bg-red-900/30 border border-red-900/50 text-red-200'
                  : 'bg-gray-800 text-gray-100'
              }`}>
                <p className="text-sm leading-relaxed whitespace-pre-wrap">
                  {message.content}
                </p>
                
                {/* Confidence Score */}
                {message.confidence && (
                  <div className="mt-2 flex items-center gap-2">
                    <span className="text-xs text-gray-400">Confidence:</span>
                    <div className="flex-1 bg-gray-700 rounded-full h-1">
                      <div
                        className={`h-1 rounded-full ${
                          message.confidence > 0.8 ? 'bg-green-500' :
                          message.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${message.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400">
                      {Math.round(message.confidence * 100)}%
                    </span>
                  </div>
                )}
                
                {/* Sources */}
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-gray-700">
                    <p className="text-xs text-gray-400 mb-1">Sources:</p>
                    <div className="space-y-1">
                      {message.sources.map((source, index) => (
                        <p key={index} className="text-xs text-blue-300">
                          • {source}
                        </p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              
              {/* Message Actions */}
              {message.type === 'bot' && (
                <div className="flex items-center gap-2 mt-2">
                  <button
                    onClick={() => copyToClipboard(message.content)}
                    className="p-1 text-gray-400 hover:text-white transition-colors"
                    title="Copy response"
                  >
                    <Copy size={14} />
                  </button>
                  <button
                    onClick={() => rateResponse(message.id, 'up')}
                    className={`p-1 transition-colors ${
                      message.userRating === 'up' 
                        ? 'text-green-400' 
                        : 'text-gray-400 hover:text-green-400'
                    }`}
                    title="Helpful"
                  >
                    <ThumbsUp size={14} />
                  </button>
                  <button
                    onClick={() => rateResponse(message.id, 'down')}
                    className={`p-1 transition-colors ${
                      message.userRating === 'down' 
                        ? 'text-red-400' 
                        : 'text-gray-400 hover:text-red-400'
                    }`}
                    title="Not helpful"
                  >
                    <ThumbsDown size={14} />
                  </button>
                  <span className="text-xs text-gray-500">
                    {formatTimestamp(message.timestamp)}
                  </span>
                </div>
              )}
              
              {/* Related Questions */}
              {message.relatedQuestions && message.relatedQuestions.length > 0 && (
                <div className="mt-3 space-y-1">
                  <p className="text-xs text-gray-400 flex items-center gap-1">
                    <Lightbulb size={12} />
                    Related questions:
                  </p>
                  {message.relatedQuestions.slice(0, 3).map((question, index) => (
                    <button
                      key={index}
                      onClick={() => handleSubmit(question)}
                      className="block w-full text-left text-xs text-blue-300 hover:text-blue-200 p-2 rounded bg-blue-900/20 hover:bg-blue-900/30 transition-colors"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              )}
              
              {/* Initial Suggestions */}
              {message.suggestions && (
                <div className="mt-3 space-y-1">
                  <p className="text-xs text-gray-400 flex items-center gap-1">
                    <Lightbulb size={12} />
                    Try asking:
                  </p>
                  {message.suggestions.map((suggestion, index) => (
                    <button
                      key={index}
                      onClick={() => handleSubmit(suggestion)}
                      className="block w-full text-left text-xs text-blue-300 hover:text-blue-200 p-2 rounded bg-blue-900/20 hover:bg-blue-900/30 transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </div>
            
            {message.type === 'user' && (
              <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center flex-shrink-0">
                <User size={16} />
              </div>
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
              <Bot size={16} />
            </div>
            <div className="bg-gray-800 p-3 rounded-lg">
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500" />
                <span className="text-sm text-gray-400">Analyzing document...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions */}
      {suggestedQuestions.length > 0 && (
        <div className="p-4 border-t border-white/10">
          <p className="text-xs text-gray-400 mb-2 flex items-center gap-1">
            <Lightbulb size={12} />
            Suggested questions:
          </p>
          <div className="flex flex-wrap gap-2">
            {suggestedQuestions.slice(0, 3).map((question, index) => (
              <button
                key={index}
                onClick={() => handleSubmit(question)}
                disabled={isLoading}
                className="text-xs px-3 py-1 rounded-full bg-blue-900/30 text-blue-300 hover:bg-blue-900/50 transition-colors disabled:opacity-50"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t border-white/10">
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={currentQuestion}
            onChange={(e) => setCurrentQuestion(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about the document..."
            className="flex-1 px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none resize-none"
            rows={1}
            disabled={isLoading}
          />
          <button
            onClick={() => handleSubmit()}
            disabled={!currentQuestion.trim() || isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}

export default InteractiveQA