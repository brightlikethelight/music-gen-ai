'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FireIcon,
  ClockIcon,
  HeartIcon,
  ChatBubbleLeftIcon,
  ShareIcon,
  UserGroupIcon,
  MusicalNoteIcon,
  FunnelIcon,
  MagnifyingGlassIcon,
} from '@heroicons/react/24/outline'
import { HeartIcon as HeartSolidIcon } from '@heroicons/react/24/solid'

import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { Avatar } from '@/components/ui/Avatar'
import { Badge } from '@/components/ui/Badge'
import { TrackCard } from '@/components/community/TrackCard'
import { UserCard } from '@/components/community/UserCard'
import { PlaylistCard } from '@/components/community/PlaylistCard'
import { TrendingTopics } from '@/components/community/TrendingTopics'
import { CommunityStats } from '@/components/community/CommunityStats'
import { useCommunity } from '@/hooks/useCommunity'

const filterOptions = [
  { value: 'all', label: 'All Content' },
  { value: 'tracks', label: 'Tracks' },
  { value: 'playlists', label: 'Playlists' },
  { value: 'users', label: 'Users' },
]

const sortOptions = [
  { value: 'trending', label: 'Trending' },
  { value: 'recent', label: 'Most Recent' },
  { value: 'popular', label: 'Most Popular' },
  { value: 'liked', label: 'Most Liked' },
]

const genreFilters = [
  { value: 'all', label: 'All Genres' },
  { value: 'pop', label: 'Pop' },
  { value: 'rock', label: 'Rock' },
  { value: 'electronic', label: 'Electronic' },
  { value: 'jazz', label: 'Jazz' },
  { value: 'classical', label: 'Classical' },
  { value: 'hip-hop', label: 'Hip-Hop' },
  { value: 'ambient', label: 'Ambient' },
]

export function CommunityHub() {
  const [activeTab, setActiveTab] = useState('trending')
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState('all')
  const [sortBy, setSortBy] = useState('trending')
  const [genreFilter, setGenreFilter] = useState('all')
  const [showFilters, setShowFilters] = useState(false)

  const {
    trendingTracks,
    featuredUsers,
    recentPlaylists,
    communityStats,
    isLoading,
    searchContent,
    likeTrack,
    followUser,
    createPlaylist,
  } = useCommunity()

  const tabs = [
    { id: 'trending', label: 'Trending', icon: FireIcon },
    { id: 'recent', label: 'Recent', icon: ClockIcon },
    { id: 'following', label: 'Following', icon: UserGroupIcon },
  ]

  const handleSearch = (query: string) => {
    setSearchQuery(query)
    if (query.trim()) {
      searchContent(query, { type: filterType, genre: genreFilter, sort: sortBy })
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex flex-col space-y-4 sm:flex-row sm:items-center sm:justify-between sm:space-y-0">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Community Hub</h1>
            <p className="text-gray-600">Discover and share amazing AI-generated music</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <Button variant="outline" onClick={() => setShowFilters(!showFilters)}>
              <FunnelIcon className="mr-2 h-4 w-4" />
              Filters
            </Button>
            <Button>
              <MusicalNoteIcon className="mr-2 h-4 w-4" />
              Share Track
            </Button>
          </div>
        </div>

        {/* Search Bar */}
        <div className="mt-6 flex space-x-4">
          <div className="flex-1">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
              <Input
                placeholder="Search tracks, users, or playlists..."
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>
        </div>

        {/* Filters */}
        <AnimatePresence>
          {showFilters && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-4"
            >
              <Select
                value={filterType}
                onChange={setFilterType}
                options={filterOptions}
                placeholder="Content Type"
              />
              <Select
                value={sortBy}
                onChange={setSortBy}
                options={sortOptions}
                placeholder="Sort By"
              />
              <Select
                value={genreFilter}
                onChange={setGenreFilter}
                options={genreFilters}
                placeholder="Genre"
              />
              <Button variant="outline" onClick={() => {
                setFilterType('all')
                setSortBy('trending')
                setGenreFilter('all')
                setSearchQuery('')
              }}>
                Clear Filters
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* Main Content */}
        <div className="lg:col-span-8">
          {/* Navigation Tabs */}
          <div className="mb-6 flex space-x-1 rounded-lg bg-gray-100 p-1">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex flex-1 items-center justify-center space-x-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{tab.label}</span>
                </button>
              )
            })}
          </div>

          {/* Content Grid */}
          <div className="space-y-6">
            <AnimatePresence mode="wait">
              {activeTab === 'trending' && (
                <motion.div
                  key="trending"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-4"
                >
                  <h2 className="text-xl font-semibold text-gray-900">Trending This Week</h2>
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    {trendingTracks.map((track, index) => (
                      <TrackCard
                        key={track.id}
                        track={track}
                        rank={index + 1}
                        onLike={() => likeTrack(track.id)}
                        onShare={() => {/* Share functionality */}}
                        showRank
                      />
                    ))}
                  </div>
                </motion.div>
              )}

              {activeTab === 'recent' && (
                <motion.div
                  key="recent"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-4"
                >
                  <h2 className="text-xl font-semibold text-gray-900">Latest Creations</h2>
                  <div className="space-y-4">
                    {trendingTracks.map((track) => (
                      <TrackCard
                        key={track.id}
                        track={track}
                        onLike={() => likeTrack(track.id)}
                        onShare={() => {/* Share functionality */}}
                        layout="horizontal"
                      />
                    ))}
                  </div>
                </motion.div>
              )}

              {activeTab === 'following' && (
                <motion.div
                  key="following"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-4"
                >
                  <h2 className="text-xl font-semibold text-gray-900">From People You Follow</h2>
                  <div className="space-y-4">
                    {trendingTracks.slice(0, 5).map((track) => (
                      <TrackCard
                        key={track.id}
                        track={track}
                        onLike={() => likeTrack(track.id)}
                        onShare={() => {/* Share functionality */}}
                        layout="horizontal"
                        showFollowButton
                      />
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Sidebar */}
        <div className="lg:col-span-4">
          <div className="space-y-6">
            {/* Community Stats */}
            <CommunityStats stats={communityStats} />

            {/* Trending Topics */}
            <TrendingTopics />

            {/* Featured Users */}
            <Card className="p-4">
              <h3 className="mb-4 text-lg font-medium text-gray-900">Featured Creators</h3>
              <div className="space-y-3">
                {featuredUsers.map((user) => (
                  <UserCard
                    key={user.id}
                    user={user}
                    onFollow={() => followUser(user.id)}
                    compact
                  />
                ))}
              </div>
            </Card>

            {/* Popular Playlists */}
            <Card className="p-4">
              <h3 className="mb-4 text-lg font-medium text-gray-900">Popular Playlists</h3>
              <div className="space-y-3">
                {recentPlaylists.map((playlist) => (
                  <PlaylistCard
                    key={playlist.id}
                    playlist={playlist}
                    compact
                  />
                ))}
              </div>
            </Card>

            {/* Quick Actions */}
            <Card className="p-4">
              <h3 className="mb-4 text-lg font-medium text-gray-900">Quick Actions</h3>
              <div className="space-y-2">
                <Button variant="outline" className="w-full justify-start">
                  <MusicalNoteIcon className="mr-2 h-4 w-4" />
                  Create Playlist
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <UserGroupIcon className="mr-2 h-4 w-4" />
                  Find Friends
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <ShareIcon className="mr-2 h-4 w-4" />
                  Share Collection
                </Button>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}