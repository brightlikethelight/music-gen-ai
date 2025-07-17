'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  UserIcon,
  BellIcon,
  ShieldCheckIcon,
  GlobeAltIcon,
  MusicalNoteIcon,
  CreditCardIcon,
  KeyIcon,
  TrashIcon,
  EyeIcon,
  EyeSlashIcon,
} from '@heroicons/react/24/outline'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Textarea } from '@/components/ui/Textarea'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { useAuth } from '@/contexts/AuthContext'
import { passwordService } from '@/lib/auth'

type SettingsTab = 'profile' | 'notifications' | 'privacy' | 'generation' | 'billing' | 'security' | 'account'

export function ProfileSettings() {
  const { user, updateUser, deleteAccount } = useAuth()
  const [activeTab, setActiveTab] = useState<SettingsTab>('profile')
  const [isLoading, setIsLoading] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  })
  const [showPasswords, setShowPasswords] = useState({
    current: false,
    new: false,
    confirm: false,
  })

  const tabs = [
    { id: 'profile', label: 'Profile', icon: UserIcon },
    { id: 'notifications', label: 'Notifications', icon: BellIcon },
    { id: 'privacy', label: 'Privacy', icon: ShieldCheckIcon },
    { id: 'generation', label: 'Generation', icon: MusicalNoteIcon },
    { id: 'billing', label: 'Billing', icon: CreditCardIcon },
    { id: 'security', label: 'Security', icon: KeyIcon },
    { id: 'account', label: 'Account', icon: TrashIcon },
  ] as const

  if (!user) return null

  const handleUpdateNotifications = async (notifications: any) => {
    setIsLoading(true)
    try {
      await updateUser({
        preferences: {
          ...user.preferences,
          notifications,
        },
      })
    } catch (error) {
      console.error('Failed to update notifications:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleUpdatePrivacy = async (privacy: any) => {
    setIsLoading(true)
    try {
      await updateUser({
        preferences: {
          ...user.preferences,
          privacy,
        },
      })
    } catch (error) {
      console.error('Failed to update privacy:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleUpdateGeneration = async (generation: any) => {
    setIsLoading(true)
    try {
      await updateUser({
        preferences: {
          ...user.preferences,
          generation,
        },
      })
    } catch (error) {
      console.error('Failed to update generation preferences:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handlePasswordChange = async () => {
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      alert('New passwords do not match')
      return
    }

    const strength = passwordService.checkPasswordStrength(passwordData.newPassword)
    if (!strength.isStrong) {
      alert('Please choose a stronger password')
      return
    }

    setIsLoading(true)
    try {
      const response = await fetch('/api/user/change-password', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          currentPassword: passwordData.currentPassword,
          newPassword: passwordData.newPassword,
        }),
      })

      if (response.ok) {
        setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' })
        alert('Password updated successfully')
      } else {
        const error = await response.json()
        alert(error.message || 'Failed to update password')
      }
    } catch (error) {
      console.error('Password change failed:', error)
      alert('Failed to update password')
    } finally {
      setIsLoading(false)
    }
  }

  const handleDeleteAccount = async () => {
    if (!showDeleteConfirm) {
      setShowDeleteConfirm(true)
      return
    }

    try {
      await deleteAccount()
    } catch (error) {
      console.error('Account deletion failed:', error)
      alert('Failed to delete account')
    }
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'profile':
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  First Name
                </label>
                <Input
                  value={user.firstName || ''}
                  onChange={(e) => updateUser({ firstName: e.target.value })}
                  className="bg-gray-700 border-gray-600 text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Last Name
                </label>
                <Input
                  value={user.lastName || ''}
                  onChange={(e) => updateUser({ lastName: e.target.value })}
                  className="bg-gray-700 border-gray-600 text-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Username
              </label>
              <Input
                value={user.username}
                onChange={(e) => updateUser({ username: e.target.value })}
                className="bg-gray-700 border-gray-600 text-white"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Email
              </label>
              <Input
                value={user.email}
                disabled
                className="bg-gray-600 border-gray-600 text-gray-400 cursor-not-allowed"
              />
              <p className="text-xs text-gray-400 mt-1">
                Email cannot be changed. Contact support if needed.
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Bio
              </label>
              <Textarea
                placeholder="Tell us about yourself..."
                className="bg-gray-700 border-gray-600 text-white"
                rows={4}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Location
                </label>
                <Input
                  placeholder="San Francisco, CA"
                  className="bg-gray-700 border-gray-600 text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Website
                </label>
                <Input
                  placeholder="https://yourwebsite.com"
                  className="bg-gray-700 border-gray-600 text-white"
                />
              </div>
            </div>
          </div>
        )

      case 'notifications':
        return (
          <div className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-white font-medium">Email Notifications</h4>
                  <p className="text-sm text-gray-400">Receive notifications via email</p>
                </div>
                <input
                  type="checkbox"
                  checked={user.preferences.notifications.email}
                  onChange={(e) => handleUpdateNotifications({
                    ...user.preferences.notifications,
                    email: e.target.checked,
                  })}
                  className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-white font-medium">Push Notifications</h4>
                  <p className="text-sm text-gray-400">Receive push notifications in browser</p>
                </div>
                <input
                  type="checkbox"
                  checked={user.preferences.notifications.push}
                  onChange={(e) => handleUpdateNotifications({
                    ...user.preferences.notifications,
                    push: e.target.checked,
                  })}
                  className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-white font-medium">Marketing Communications</h4>
                  <p className="text-sm text-gray-400">Receive product updates and promotions</p>
                </div>
                <input
                  type="checkbox"
                  checked={user.preferences.notifications.marketing}
                  onChange={(e) => handleUpdateNotifications({
                    ...user.preferences.notifications,
                    marketing: e.target.checked,
                  })}
                  className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                />
              </div>
            </div>

            <div className="border-t border-gray-700 pt-6">
              <h3 className="text-lg font-semibold text-white mb-4">Notification Types</h3>
              <div className="space-y-3">
                {[
                  'New followers',
                  'Likes on your tracks',
                  'Comments on your tracks',
                  'Collaboration requests',
                  'Generation completed',
                  'System updates',
                ].map((type) => (
                  <div key={type} className="flex items-center justify-between">
                    <span className="text-gray-300">{type}</span>
                    <input
                      type="checkbox"
                      defaultChecked
                      className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )

      case 'privacy':
        return (
          <div className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-white font-medium">Public Profile</h4>
                  <p className="text-sm text-gray-400">Make your profile visible to everyone</p>
                </div>
                <input
                  type="checkbox"
                  checked={user.preferences.privacy.profilePublic}
                  onChange={(e) => handleUpdatePrivacy({
                    ...user.preferences.privacy,
                    profilePublic: e.target.checked,
                  })}
                  className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-white font-medium">Public Generations</h4>
                  <p className="text-sm text-gray-400">Show your music generations publicly</p>
                </div>
                <input
                  type="checkbox"
                  checked={user.preferences.privacy.generationsPublic}
                  onChange={(e) => handleUpdatePrivacy({
                    ...user.preferences.privacy,
                    generationsPublic: e.target.checked,
                  })}
                  className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-white font-medium">Allow Following</h4>
                  <p className="text-sm text-gray-400">Let other users follow you</p>
                </div>
                <input
                  type="checkbox"
                  checked={user.preferences.privacy.allowFollowing}
                  onChange={(e) => handleUpdatePrivacy({
                    ...user.preferences.privacy,
                    allowFollowing: e.target.checked,
                  })}
                  className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                />
              </div>
            </div>

            <div className="border-t border-gray-700 pt-6">
              <h3 className="text-lg font-semibold text-white mb-4">Data & Analytics</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Usage Analytics</span>
                  <input
                    type="checkbox"
                    defaultChecked
                    className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Performance Monitoring</span>
                  <input
                    type="checkbox"
                    defaultChecked
                    className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Crash Reports</span>
                  <input
                    type="checkbox"
                    defaultChecked
                    className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                  />
                </div>
              </div>
            </div>
          </div>
        )

      case 'generation':
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Default Genre
                </label>
                <Select
                  value={user.preferences.generation.defaultGenre}
                  onValueChange={(value) => handleUpdateGeneration({
                    ...user.preferences.generation,
                    defaultGenre: value,
                  })}
                >
                  <SelectTrigger className="bg-gray-700 border-gray-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="electronic">Electronic</SelectItem>
                    <SelectItem value="pop">Pop</SelectItem>
                    <SelectItem value="rock">Rock</SelectItem>
                    <SelectItem value="jazz">Jazz</SelectItem>
                    <SelectItem value="classical">Classical</SelectItem>
                    <SelectItem value="ambient">Ambient</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Default Duration (seconds)
                </label>
                <Input
                  type="number"
                  min="10"
                  max="180"
                  value={user.preferences.generation.defaultDuration}
                  onChange={(e) => handleUpdateGeneration({
                    ...user.preferences.generation,
                    defaultDuration: parseInt(e.target.value),
                  })}
                  className="bg-gray-700 border-gray-600 text-white"
                />
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-white font-medium">Auto-Save Generations</h4>
                <p className="text-sm text-gray-400">Automatically save generated tracks</p>
              </div>
              <input
                type="checkbox"
                checked={user.preferences.generation.autoSave}
                onChange={(e) => handleUpdateGeneration({
                  ...user.preferences.generation,
                  autoSave: e.target.checked,
                })}
                className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
              />
            </div>

            <div className="border-t border-gray-700 pt-6">
              <h3 className="text-lg font-semibold text-white mb-4">Quality Settings</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Audio Quality
                  </label>
                  <Select defaultValue="high">
                    <SelectTrigger className="bg-gray-700 border-gray-600 text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low (128 kbps)</SelectItem>
                      <SelectItem value="medium">Medium (256 kbps)</SelectItem>
                      <SelectItem value="high">High (320 kbps)</SelectItem>
                      <SelectItem value="lossless">Lossless (WAV)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Sample Rate
                  </label>
                  <Select defaultValue="44100">
                    <SelectTrigger className="bg-gray-700 border-gray-600 text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="22050">22.05 kHz</SelectItem>
                      <SelectItem value="44100">44.1 kHz</SelectItem>
                      <SelectItem value="48000">48 kHz</SelectItem>
                      <SelectItem value="96000">96 kHz</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </div>
        )

      case 'security':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Change Password</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Current Password
                  </label>
                  <div className="relative">
                    <Input
                      type={showPasswords.current ? 'text' : 'password'}
                      value={passwordData.currentPassword}
                      onChange={(e) => setPasswordData(prev => ({ ...prev, currentPassword: e.target.value }))}
                      className="bg-gray-700 border-gray-600 text-white pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPasswords(prev => ({ ...prev, current: !prev.current }))}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                    >
                      {showPasswords.current ? <EyeSlashIcon className="w-5 h-5" /> : <EyeIcon className="w-5 h-5" />}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    New Password
                  </label>
                  <div className="relative">
                    <Input
                      type={showPasswords.new ? 'text' : 'password'}
                      value={passwordData.newPassword}
                      onChange={(e) => setPasswordData(prev => ({ ...prev, newPassword: e.target.value }))}
                      className="bg-gray-700 border-gray-600 text-white pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPasswords(prev => ({ ...prev, new: !prev.new }))}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                    >
                      {showPasswords.new ? <EyeSlashIcon className="w-5 h-5" /> : <EyeIcon className="w-5 h-5" />}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Confirm New Password
                  </label>
                  <div className="relative">
                    <Input
                      type={showPasswords.confirm ? 'text' : 'password'}
                      value={passwordData.confirmPassword}
                      onChange={(e) => setPasswordData(prev => ({ ...prev, confirmPassword: e.target.value }))}
                      className="bg-gray-700 border-gray-600 text-white pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPasswords(prev => ({ ...prev, confirm: !prev.confirm }))}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                    >
                      {showPasswords.confirm ? <EyeSlashIcon className="w-5 h-5" /> : <EyeIcon className="w-5 h-5" />}
                    </button>
                  </div>
                </div>

                <Button
                  onClick={handlePasswordChange}
                  disabled={isLoading || !passwordData.currentPassword || !passwordData.newPassword || !passwordData.confirmPassword}
                  className="bg-purple-600 hover:bg-purple-700"
                >
                  Update Password
                </Button>
              </div>
            </div>

            <div className="border-t border-gray-700 pt-6">
              <h3 className="text-lg font-semibold text-white mb-4">Two-Factor Authentication</h3>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-300">Secure your account with 2FA</p>
                  <p className="text-sm text-gray-400">Currently disabled</p>
                </div>
                <Button variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-700">
                  Enable 2FA
                </Button>
              </div>
            </div>

            <div className="border-t border-gray-700 pt-6">
              <h3 className="text-lg font-semibold text-white mb-4">Active Sessions</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                  <div>
                    <p className="text-white">Current Session</p>
                    <p className="text-sm text-gray-400">Chrome on macOS • San Francisco, CA</p>
                  </div>
                  <span className="text-xs text-green-400 bg-green-900 px-2 py-1 rounded">Active</span>
                </div>
              </div>
            </div>
          </div>
        )

      case 'account':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Export Data</h3>
              <p className="text-gray-400 mb-4">
                Download a copy of all your data including tracks, projects, and settings.
              </p>
              <Button variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-700">
                Request Data Export
              </Button>
            </div>

            <div className="border-t border-gray-700 pt-6">
              <h3 className="text-lg font-semibold text-red-400 mb-4">Danger Zone</h3>
              <div className="space-y-4">
                <div className="p-4 border border-red-700 rounded-lg bg-red-900/20">
                  <h4 className="text-red-400 font-medium mb-2">Delete Account</h4>
                  <p className="text-gray-300 text-sm mb-4">
                    Permanently delete your account and all associated data. This action cannot be undone.
                  </p>
                  
                  {showDeleteConfirm ? (
                    <div className="space-y-3">
                      <p className="text-red-300 text-sm font-medium">
                        Are you absolutely sure? This will permanently delete your account and all your data.
                      </p>
                      <div className="flex space-x-3">
                        <Button
                          onClick={handleDeleteAccount}
                          className="bg-red-600 hover:bg-red-700 text-white"
                        >
                          Yes, Delete My Account
                        </Button>
                        <Button
                          onClick={() => setShowDeleteConfirm(false)}
                          variant="outline"
                          className="border-gray-600 text-gray-300 hover:bg-gray-700"
                        >
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <Button
                      onClick={() => setShowDeleteConfirm(true)}
                      variant="outline"
                      className="border-red-600 text-red-400 hover:bg-red-900"
                    >
                      <TrashIcon className="w-4 h-4 mr-2" />
                      Delete Account
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Settings</h1>
        <p className="text-gray-400">Manage your account preferences and settings</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Sidebar */}
        <div className="lg:col-span-1">
          <nav className="space-y-1">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'bg-purple-600 text-white'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{tab.label}</span>
                </button>
              )
            })}
          </nav>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                {tabs.find(tab => tab.id === activeTab)?.icon && (
                  <tabs.find(tab => tab.id === activeTab)!.icon className="w-5 h-5 mr-2" />
                )}
                {tabs.find(tab => tab.id === activeTab)?.label}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {renderTabContent()}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}