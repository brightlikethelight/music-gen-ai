'use client'

import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import {
  UserCircleIcon,
  CameraIcon,
  PencilIcon,
  CheckIcon,
  XMarkIcon,
  MusicalNoteIcon,
  HeartIcon,
  UserGroupIcon,
  CalendarIcon,
  GlobeAltIcon,
  LockClosedIcon,
  BellIcon,
} from '@heroicons/react/24/outline'
import { HeartIcon as HeartSolidIcon } from '@heroicons/react/24/solid'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Textarea } from '@/components/ui/Textarea'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { useAuth } from '@/contexts/AuthContext'
import { User } from '@/lib/auth'

interface UserProfileProps {
  isOwnProfile?: boolean
}

export function UserProfile({ isOwnProfile = true }: UserProfileProps) {
  const { user, updateUser } = useAuth()
  const [isEditing, setIsEditing] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isFollowing, setIsFollowing] = useState(false)
  const [editForm, setEditForm] = useState({
    firstName: user?.firstName || '',
    lastName: user?.lastName || '',
    username: user?.username || '',
    bio: '',
    location: '',
    website: '',
  })

  const fileInputRef = useRef<HTMLInputElement>(null)

  if (!user) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-500"></div>
      </div>
    )
  }

  const handleSaveProfile = async () => {
    try {
      await updateUser({
        firstName: editForm.firstName,
        lastName: editForm.lastName,
        username: editForm.username,
      })
      setIsEditing(false)
    } catch (error) {
      console.error('Failed to update profile:', error)
    }
  }

  const handleAvatarUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    try {
      const formData = new FormData()
      formData.append('avatar', file)

      const response = await fetch('/api/user/avatar', {
        method: 'POST',
        body: formData,
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
        },
      })

      if (response.ok) {
        const { avatarUrl } = await response.json()
        await updateUser({ avatar: avatarUrl })
      }
    } catch (error) {
      console.error('Avatar upload failed:', error)
    } finally {
      setIsUploading(false)
    }
  }

  const handleFollow = async () => {
    try {
      const response = await fetch(`/api/user/${user.id}/follow`, {
        method: isFollowing ? 'DELETE' : 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      })

      if (response.ok) {
        setIsFollowing(!isFollowing)
      }
    } catch (error) {
      console.error('Follow action failed:', error)
    }
  }

  const formatJoinDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
    })
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Profile Header */}
      <Card className="bg-gray-800 border-gray-700">
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row lg:items-start lg:space-x-8 space-y-6 lg:space-y-0">
            {/* Avatar Section */}
            <div className="flex flex-col items-center lg:items-start space-y-4">
              <div className="relative">
                <div className="w-32 h-32 rounded-full overflow-hidden bg-gray-700 flex items-center justify-center">
                  {user.avatar ? (
                    <img
                      src={user.avatar}
                      alt={`${user.username}'s avatar`}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <UserCircleIcon className="w-20 h-20 text-gray-400" />
                  )}
                </div>
                
                {isOwnProfile && (
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                    className="absolute bottom-0 right-0 bg-purple-600 hover:bg-purple-700 text-white p-2 rounded-full shadow-lg transition-colors"
                  >
                    {isUploading ? (
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <CameraIcon className="w-4 h-4" />
                    )}
                  </button>
                )}
                
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleAvatarUpload}
                  className="hidden"
                />
              </div>

              {/* Tier Badge */}
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                user.tier === 'enterprise' ? 'bg-purple-900 text-purple-200' :
                user.tier === 'pro' ? 'bg-blue-900 text-blue-200' :
                'bg-gray-700 text-gray-300'
              }`}>
                {user.tier.charAt(0).toUpperCase() + user.tier.slice(1)}
              </div>
            </div>

            {/* Profile Info */}
            <div className="flex-1 space-y-4">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                <div className="space-y-2">
                  {isEditing ? (
                    <div className="space-y-3">
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                        <Input
                          value={editForm.firstName}
                          onChange={(e) => setEditForm(prev => ({ ...prev, firstName: e.target.value }))}
                          placeholder="First name"
                          className="bg-gray-700 border-gray-600 text-white"
                        />
                        <Input
                          value={editForm.lastName}
                          onChange={(e) => setEditForm(prev => ({ ...prev, lastName: e.target.value }))}
                          placeholder="Last name"
                          className="bg-gray-700 border-gray-600 text-white"
                        />
                      </div>
                      <Input
                        value={editForm.username}
                        onChange={(e) => setEditForm(prev => ({ ...prev, username: e.target.value }))}
                        placeholder="Username"
                        className="bg-gray-700 border-gray-600 text-white"
                      />
                    </div>
                  ) : (
                    <>
                      <h1 className="text-2xl lg:text-3xl font-bold text-white">
                        {user.firstName && user.lastName 
                          ? `${user.firstName} ${user.lastName}`
                          : user.username
                        }
                      </h1>
                      <p className="text-gray-400">@{user.username}</p>
                    </>
                  )}
                </div>

                <div className="flex items-center space-x-3">
                  {isOwnProfile ? (
                    isEditing ? (
                      <div className="flex space-x-2">
                        <Button
                          size="sm"
                          onClick={handleSaveProfile}
                          className="bg-green-600 hover:bg-green-700"
                        >
                          <CheckIcon className="w-4 h-4 mr-1" />
                          Save
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setIsEditing(false)}
                          className="border-gray-600 text-gray-300 hover:bg-gray-700"
                        >
                          <XMarkIcon className="w-4 h-4 mr-1" />
                          Cancel
                        </Button>
                      </div>
                    ) : (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setIsEditing(true)}
                        className="border-gray-600 text-gray-300 hover:bg-gray-700"
                      >
                        <PencilIcon className="w-4 h-4 mr-2" />
                        Edit Profile
                      </Button>
                    )
                  ) : (
                    <div className="flex space-x-2">
                      <Button
                        onClick={handleFollow}
                        className={isFollowing 
                          ? "bg-gray-600 hover:bg-gray-700 text-white"
                          : "bg-purple-600 hover:bg-purple-700 text-white"
                        }
                      >
                        <UserGroupIcon className="w-4 h-4 mr-2" />
                        {isFollowing ? 'Following' : 'Follow'}
                      </Button>
                      <Button
                        variant="outline"
                        className="border-gray-600 text-gray-300 hover:bg-gray-700"
                      >
                        <MusicalNoteIcon className="w-4 h-4 mr-2" />
                        Collaborate
                      </Button>
                    </div>
                  )}
                </div>
              </div>

              {/* Profile Stats */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="text-center lg:text-left">
                  <div className="text-2xl font-bold text-white">127</div>
                  <div className="text-sm text-gray-400">Generations</div>
                </div>
                <div className="text-center lg:text-left">
                  <div className="text-2xl font-bold text-white">23</div>
                  <div className="text-sm text-gray-400">Projects</div>
                </div>
                <div className="text-center lg:text-left">
                  <div className="text-2xl font-bold text-white">1.2K</div>
                  <div className="text-sm text-gray-400">Followers</div>
                </div>
                <div className="text-center lg:text-left">
                  <div className="text-2xl font-bold text-white">834</div>
                  <div className="text-sm text-gray-400">Following</div>
                </div>
              </div>

              {/* Bio and Details */}
              <div className="space-y-3">
                <p className="text-gray-300">
                  🎵 Creating AI-generated music since 2024. Electronic & ambient producer. 
                  Always experimenting with new sounds and collaborating with fellow creators.
                </p>
                
                <div className="flex flex-wrap items-center gap-4 text-sm text-gray-400">
                  <div className="flex items-center space-x-1">
                    <CalendarIcon className="w-4 h-4" />
                    <span>Joined {formatJoinDate(user.createdAt)}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <GlobeAltIcon className="w-4 h-4" />
                    <span>San Francisco, CA</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    {user.preferences.privacy.profilePublic ? (
                      <>
                        <GlobeAltIcon className="w-4 h-4 text-green-400" />
                        <span className="text-green-400">Public</span>
                      </>
                    ) : (
                      <>
                        <LockClosedIcon className="w-4 h-4 text-yellow-400" />
                        <span className="text-yellow-400">Private</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Profile Content Tabs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Recent Generations */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <MusicalNoteIcon className="w-5 h-5 mr-2" />
                Recent Generations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[1, 2, 3].map((item) => (
                  <div key={item} className="flex items-center space-x-4 p-4 bg-gray-700 rounded-lg">
                    <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                      <MusicalNoteIcon className="w-6 h-6 text-white" />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-medium text-white">Cosmic Journey {item}</h4>
                      <p className="text-sm text-gray-400">Electronic • 2:34 • 2 days ago</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button className="text-gray-400 hover:text-red-400 transition-colors">
                        <HeartIcon className="w-5 h-5" />
                      </button>
                      <span className="text-sm text-gray-400">24</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Activity Feed */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Activity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-purple-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <p className="text-white">Created a new track: "Midnight Waves"</p>
                    <p className="text-sm text-gray-400">3 hours ago</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <p className="text-white">Followed @alex_composer</p>
                    <p className="text-sm text-gray-400">1 day ago</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <p className="text-white">Liked "Jazz Fusion Experiment" by @sarah_beats</p>
                    <p className="text-sm text-gray-400">2 days ago</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Preferences Summary */}
          {isOwnProfile && (
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Preferences</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Notifications</span>
                  <div className="flex items-center space-x-1">
                    <BellIcon className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-400">
                      {user.preferences.notifications.email ? 'On' : 'Off'}
                    </span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Profile Visibility</span>
                  <div className="flex items-center space-x-1">
                    {user.preferences.privacy.profilePublic ? (
                      <GlobeAltIcon className="w-4 h-4 text-green-400" />
                    ) : (
                      <LockClosedIcon className="w-4 h-4 text-yellow-400" />
                    )}
                    <span className="text-sm text-gray-400">
                      {user.preferences.privacy.profilePublic ? 'Public' : 'Private'}
                    </span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Default Genre</span>
                  <span className="text-sm text-gray-400 capitalize">
                    {user.preferences.generation.defaultGenre}
                  </span>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Popular Tracks */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Popular Tracks</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[1, 2, 3].map((item) => (
                  <div key={item} className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-cyan-500 rounded flex items-center justify-center text-white text-sm font-bold">
                      {item}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-white">Ambient Dreams</p>
                      <p className="text-xs text-gray-400">1.2K plays</p>
                    </div>
                    <HeartSolidIcon className="w-4 h-4 text-red-400" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Achievements */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Achievements</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-3">
                <div className="text-center">
                  <div className="w-12 h-12 bg-yellow-600 rounded-full flex items-center justify-center mx-auto mb-2">
                    🏆
                  </div>
                  <p className="text-xs text-gray-400">Top Creator</p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-2">
                    🎵
                  </div>
                  <p className="text-xs text-gray-400">100 Tracks</p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-2">
                    ❤️
                  </div>
                  <p className="text-xs text-gray-400">1K Likes</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}