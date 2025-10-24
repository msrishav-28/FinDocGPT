import React, { useState, useEffect, useCallback } from 'react'
import { Plus, Edit, Trash2, Bell, BellOff, Settings, Save, X } from 'lucide-react'
import { API } from '../../services'

const AlertManager = ({ userId = 'demo-user' }) => {
  const [alertRules, setAlertRules] = useState([])
  const [activeAlerts, setActiveAlerts] = useState([])
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [editingRule, setEditingRule] = useState(null)
  const [loading, setLoading] = useState(true)
  const [alertTypes, setAlertTypes] = useState({ alert_types: [], severities: [] })

  useEffect(() => {
    loadAlertRules()
    loadActiveAlerts()
    loadAlertTypes()
  }, [userId])

  const loadAlertRules = async () => {
    try {
      const response = await API.get(`/alerts/rules?user_id=${userId}`)
      setAlertRules(response.data)
    } catch (error) {
      console.error('Error loading alert rules:', error)
    }
  }

  const loadActiveAlerts = async () => {
    try {
      const response = await API.get(`/alerts/active?user_id=${userId}`)
      setActiveAlerts(response.data)
    } catch (error) {
      console.error('Error loading active alerts:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadAlertTypes = async () => {
    try {
      const response = await API.get('/alerts/types')
      setAlertTypes(response.data)
    } catch (error) {
      console.error('Error loading alert types:', error)
    }
  }

  const createAlertRule = async (ruleData) => {
    try {
      await API.post('/alerts/rules', { ...ruleData, user_id: userId })
      await loadAlertRules()
      setShowCreateModal(false)
    } catch (error) {
      console.error('Error creating alert rule:', error)
    }
  }

  const updateAlertRule = async (ruleId, updates) => {
    try {
      await API.put(`/alerts/rules/${ruleId}`, updates)
      await loadAlertRules()
      setEditingRule(null)
    } catch (error) {
      console.error('Error updating alert rule:', error)
    }
  }

  const deleteAlertRule = async (ruleId) => {
    if (window.confirm('Are you sure you want to delete this alert rule?')) {
      try {
        await API.delete(`/alerts/rules/${ruleId}`)
        await loadAlertRules()
      } catch (error) {
        console.error('Error deleting alert rule:', error)
      }
    }
  }

  const acknowledgeAlert = async (alertId) => {
    try {
      await API.post(`/alerts/acknowledge/${alertId}?user_id=${userId}`)
      await loadActiveAlerts()
    } catch (error) {
      console.error('Error acknowledging alert:', error)
    }
  }

  const setupDefaultRules = async () => {
    try {
      await API.post(`/alerts/setup-defaults/${userId}`)
      await loadAlertRules()
    } catch (error) {
      console.error('Error setting up default rules:', error)
    }
  }

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'text-red-400 bg-red-900/20 border-red-500'
      case 'high': return 'text-orange-400 bg-orange-900/20 border-orange-500'
      case 'medium': return 'text-blue-400 bg-blue-900/20 border-blue-500'
      case 'low': return 'text-green-400 bg-green-900/20 border-green-500'
      default: return 'text-gray-400 bg-gray-900/20 border-gray-500'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between p-4 rounded-xl glass">
        <div>
          <h2 className="text-lg font-semibold text-white">Alert Management</h2>
          <p className="text-sm text-gray-400">
            Manage your alert rules and view active notifications
          </p>
        </div>
        <div className="flex gap-2">
          {alertRules.length === 0 && (
            <button
              onClick={setupDefaultRules}
              className="px-3 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors text-sm"
            >
              Setup Defaults
            </button>
          )}
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center gap-2 px-3 py-2 rounded-md bg-green-600 text-white hover:bg-green-700 transition-colors"
          >
            <Plus size={16} />
            New Rule
          </button>
        </div>
      </div>

      {/* Active Alerts */}
      {activeAlerts.length > 0 && (
        <div className="p-4 rounded-xl glass">
          <h3 className="text-md font-semibold text-white mb-4">Active Alerts</h3>
          <div className="space-y-3">
            {activeAlerts.map(alert => (
              <div
                key={alert.id}
                className={`p-3 rounded-lg border ${getSeverityColor(alert.severity)}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium text-white">{alert.title}</h4>
                    <p className="text-sm text-gray-300 mt-1">{alert.message}</p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-gray-400">
                      <span>{new Date(alert.created_at).toLocaleString()}</span>
                      <span className="capitalize">{alert.type.replace('_', ' ')}</span>
                    </div>
                  </div>
                  {!alert.acknowledged && (
                    <button
                      onClick={() => acknowledgeAlert(alert.id)}
                      className="px-2 py-1 rounded text-xs bg-white/10 text-white hover:bg-white/20 transition-colors"
                    >
                      Acknowledge
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Alert Rules */}
      <div className="p-4 rounded-xl glass">
        <h3 className="text-md font-semibold text-white mb-4">Alert Rules</h3>
        
        {alertRules.length === 0 ? (
          <div className="text-center py-8">
            <Bell className="mx-auto text-gray-400 mb-2" size={32} />
            <p className="text-gray-400">No alert rules configured</p>
            <button
              onClick={setupDefaultRules}
              className="mt-2 px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors"
            >
              Setup Default Rules
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {alertRules.map(rule => (
              <AlertRuleItem
                key={rule.id}
                rule={rule}
                onEdit={setEditingRule}
                onDelete={deleteAlertRule}
                onUpdate={updateAlertRule}
                getSeverityColor={getSeverityColor}
                isEditing={editingRule?.id === rule.id}
                alertTypes={alertTypes}
              />
            ))}
          </div>
        )}
      </div>

      {/* Create Alert Modal */}
      {showCreateModal && (
        <CreateAlertModal
          onClose={() => setShowCreateModal(false)}
          onCreate={createAlertRule}
          alertTypes={alertTypes}
        />
      )}
    </div>
  )
}

const AlertRuleItem = ({ 
  rule, 
  onEdit, 
  onDelete, 
  onUpdate, 
  getSeverityColor, 
  isEditing, 
  alertTypes 
}) => {
  const [editForm, setEditForm] = useState({
    name: rule.name,
    severity: rule.severity,
    enabled: rule.enabled,
    conditions: rule.conditions
  })

  const handleSave = () => {
    onUpdate(rule.id, editForm)
  }

  const handleCancel = () => {
    setEditForm({
      name: rule.name,
      severity: rule.severity,
      enabled: rule.enabled,
      conditions: rule.conditions
    })
    onEdit(null)
  }

  if (isEditing) {
    return (
      <div className="p-4 rounded-lg border border-white/20 bg-white/5">
        <div className="space-y-3">
          <input
            type="text"
            value={editForm.name}
            onChange={(e) => setEditForm(prev => ({ ...prev, name: e.target.value }))}
            className="w-full p-2 rounded-md bg-gray-800 text-white border border-gray-700"
          />
          
          <div className="flex gap-3">
            <select
              value={editForm.severity}
              onChange={(e) => setEditForm(prev => ({ ...prev, severity: e.target.value }))}
              className="p-2 rounded-md bg-gray-800 text-white border border-gray-700"
            >
              {alertTypes.severities.map(severity => (
                <option key={severity.value} value={severity.value}>
                  {severity.label}
                </option>
              ))}
            </select>
            
            <label className="flex items-center gap-2 text-white">
              <input
                type="checkbox"
                checked={editForm.enabled}
                onChange={(e) => setEditForm(prev => ({ ...prev, enabled: e.target.checked }))}
                className="rounded"
              />
              Enabled
            </label>
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={handleSave}
              className="flex items-center gap-1 px-3 py-1 rounded-md bg-green-600 text-white hover:bg-green-700 transition-colors text-sm"
            >
              <Save size={14} />
              Save
            </button>
            <button
              onClick={handleCancel}
              className="flex items-center gap-1 px-3 py-1 rounded-md bg-gray-600 text-white hover:bg-gray-700 transition-colors text-sm"
            >
              <X size={14} />
              Cancel
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={`p-4 rounded-lg border ${getSeverityColor(rule.severity)}`}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <h4 className="font-medium text-white">{rule.name}</h4>
            <span className={`px-2 py-1 rounded-full text-xs ${getSeverityColor(rule.severity)}`}>
              {rule.severity}
            </span>
            {rule.enabled ? (
              <Bell className="text-green-400" size={16} />
            ) : (
              <BellOff className="text-gray-400" size={16} />
            )}
          </div>
          <p className="text-sm text-gray-300 mt-1 capitalize">
            {rule.type.replace('_', ' ')} alert
          </p>
          <div className="text-xs text-gray-400 mt-1">
            Created: {new Date(rule.created_at).toLocaleDateString()}
            {rule.last_triggered && (
              <span className="ml-4">
                Last triggered: {new Date(rule.last_triggered).toLocaleString()}
              </span>
            )}
          </div>
        </div>
        
        <div className="flex gap-1">
          <button
            onClick={() => onEdit(rule)}
            className="p-2 rounded text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <Edit size={16} />
          </button>
          <button
            onClick={() => onDelete(rule.id)}
            className="p-2 rounded text-gray-400 hover:text-red-400 hover:bg-red-900/20 transition-colors"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}

const CreateAlertModal = ({ onClose, onCreate, alertTypes }) => {
  const [form, setForm] = useState({
    name: '',
    type: 'price_movement',
    severity: 'medium',
    conditions: {
      price_change_percent: 5.0,
      cooldown_minutes: 30
    }
  })

  const handleSubmit = (e) => {
    e.preventDefault()
    onCreate(form)
  }

  const updateCondition = (key, value) => {
    setForm(prev => ({
      ...prev,
      conditions: {
        ...prev.conditions,
        [key]: value
      }
    }))
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-xl p-6 w-full max-w-md mx-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Create Alert Rule</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <X size={20} />
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Name</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm(prev => ({ ...prev, name: e.target.value }))}
              className="w-full p-2 rounded-md bg-gray-800 text-white border border-gray-700"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Type</label>
            <select
              value={form.type}
              onChange={(e) => setForm(prev => ({ ...prev, type: e.target.value }))}
              className="w-full p-2 rounded-md bg-gray-800 text-white border border-gray-700"
            >
              {alertTypes.alert_types.map(type => (
                <option key={type.value} value={type.value}>{type.label}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Severity</label>
            <select
              value={form.severity}
              onChange={(e) => setForm(prev => ({ ...prev, severity: e.target.value }))}
              className="w-full p-2 rounded-md bg-gray-800 text-white border border-gray-700"
            >
              {alertTypes.severities.map(severity => (
                <option key={severity.value} value={severity.value}>{severity.label}</option>
              ))}
            </select>
          </div>
          
          {form.type === 'price_movement' && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Price Change Threshold (%)
              </label>
              <input
                type="number"
                step="0.1"
                value={form.conditions.price_change_percent}
                onChange={(e) => updateCondition('price_change_percent', parseFloat(e.target.value))}
                className="w-full p-2 rounded-md bg-gray-800 text-white border border-gray-700"
              />
            </div>
          )}
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Cooldown (minutes)
            </label>
            <input
              type="number"
              value={form.conditions.cooldown_minutes}
              onChange={(e) => updateCondition('cooldown_minutes', parseInt(e.target.value))}
              className="w-full p-2 rounded-md bg-gray-800 text-white border border-gray-700"
            />
          </div>
          
          <div className="flex gap-2 pt-4">
            <button
              type="submit"
              className="flex-1 py-2 px-4 rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors"
            >
              Create Rule
            </button>
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-2 px-4 rounded-md bg-gray-600 text-white hover:bg-gray-700 transition-colors"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default AlertManager