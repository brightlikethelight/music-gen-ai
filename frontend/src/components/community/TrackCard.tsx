'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  PlayIcon,
  PauseIcon,
  HeartIcon,
  ChatBubbleLeftIcon,
  ShareIcon,
  EllipsisHorizontalIcon,
  UserPlusIcon,
} from '@heroicons/react/24/outline'
import { HeartIcon as HeartSolidIcon } from '@heroicons/react/24/solid'
import { Avatar } from '@/components/ui/Avatar'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { WaveformVisualization } from '@/components/audio/WaveformVisualization'
import { formatTime, formatNumber } from '@/utils/formatters'

interface Track {
  id: string
  title: string
  description?: string
  genre: string
  duration: number
  createdAt: string
  audioUrl: string
  waveformData: number[]
  user: {
    id: string
    name: string
    username: string
    avatar: string
    isFollowing?: boolean
  }
  stats: {
    plays: number
    likes: number
    comments: number
    shares: number
  }
  isLiked?: boolean
  isPlaying?: boolean
}

interface TrackCardProps {
  track: Track
  rank?: number
  layout?: 'vertical' | 'horizontal'
  showRank?: boolean
  showFollowButton?: boolean
  onLike?: () => void
  onShare?: () => void
  onPlay?: () => void
  onComment?: () => void
  onFollow?: () => void
}

export function TrackCard({
  track,
  rank,
  layout = 'vertical',
  showRank = false,
  showFollowButton = false,
  onLike,
  onShare,
  onPlay,
  onComment,
  onFollow,
}: TrackCardProps) {
  const [isHovered, setIsHovered] = useState(false)
  const [showMenu, setShowMenu] = useState(false)

  const handleLike = (e: React.MouseEvent) => {
    e.stopPropagation()
    onLike?.()
  }

  const handleShare = (e: React.MouseEvent) => {
    e.stopPropagation()
    onShare?.()
  }

  const handleComment = (e: React.MouseEvent) => {
    e.stopPropagation()
    onComment?.()
  }

  const handleFollow = (e: React.MouseEvent) => {
    e.stopPropagation()
    onFollow?.()
  }

  const handlePlay = (e: React.MouseEvent) => {
    e.stopPropagation()
    onPlay?.()
  }

  if (layout === 'horizontal') {
    return (
      <motion.div
        className="group relative overflow-hidden rounded-lg bg-white p-4 shadow-sm hover:shadow-md"
        onHoverStart={() => setIsHovered(true)}
        onHoverEnd={() => setIsHovered(false)}
        whileHover={{ y: -2 }}
      >
        <div className="flex items-center space-x-4">
          {/* Rank */}
          {showRank && rank && (
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-yellow-400 to-orange-500 text-sm font-bold text-white">
              {rank}
            </div>
          )}

          {/* Play Button & Waveform */}
          <div className="relative flex-shrink-0">
            <div className="h-16 w-24 overflow-hidden rounded-md bg-gray-100">
              <WaveformVisualization
                data={track.waveformData}
                isPlaying={track.isPlaying}
                height={64}
                className="h-full w-full"
                animateOnPlay
              />
            </div>
            <button
              onClick={handlePlay}
              className="absolute inset-0 flex items-center justify-center bg-black/20 opacity-0 transition-opacity group-hover:opacity-100"
            >
              {track.isPlaying ? (
                <PauseIcon className="h-6 w-6 text-white" />
              ) : (
                <PlayIcon className="h-6 w-6 text-white" />
              )}
            </button>
          </div>

          {/* Track Info */}
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between">
              <div className="min-w-0 flex-1">
                <h3 className="truncate text-lg font-medium text-gray-900">{track.title}</h3>
                <div className="mt-1 flex items-center space-x-2">
                  <Avatar src={track.user.avatar} size="sm" />
                  <span className="text-sm text-gray-600">by {track.user.name}</span>
                  <Badge variant="secondary">{track.genre}</Badge>
                </div>
                {track.description && (
                  <p className="mt-2 line-clamp-2 text-sm text-gray-500">{track.description}</p>
                )}
              </div>

              {/* Follow Button */}
              {showFollowButton && !track.user.isFollowing && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleFollow}
                  className="ml-2"
                >
                  <UserPlusIcon className="mr-1 h-4 w-4" />
                  Follow
                </Button>
              )}
            </div>

            {/* Stats & Actions */}
            <div className="mt-3 flex items-center justify-between">
              <div className="flex items-center space-x-4 text-sm text-gray-500">
                <span>{formatNumber(track.stats.plays)} plays</span>
                <span>{formatTime(track.duration)}</span>
                <span>{new Date(track.createdAt).toLocaleDateString()}</span>
              </div>

              <div className="flex items-center space-x-2">
                <button
                  onClick={handleLike}
                  className="flex items-center space-x-1 text-gray-400 hover:text-red-500"
                >
                  {track.isLiked ? (
                    <HeartSolidIcon className="h-4 w-4 text-red-500" />
                  ) : (
                    <HeartIcon className="h-4 w-4" />
                  )}
                  <span className="text-xs">{formatNumber(track.stats.likes)}</span>
                </button>

                <button
                  onClick={handleComment}
                  className="flex items-center space-x-1 text-gray-400 hover:text-blue-500"
                >
                  <ChatBubbleLeftIcon className="h-4 w-4" />
                  <span className="text-xs">{formatNumber(track.stats.comments)}</span>
                </button>

                <button
                  onClick={handleShare}
                  className="flex items-center space-x-1 text-gray-400 hover:text-green-500"
                >
                  <ShareIcon className="h-4 w-4" />
                  <span className="text-xs">{formatNumber(track.stats.shares)}</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    )
  }

  // Vertical layout
  return (
    <motion.div
      className="group relative overflow-hidden rounded-lg bg-white shadow-sm hover:shadow-lg"
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      whileHover={{ y: -4 }}
    >
      {/* Rank Badge */}
      {showRank && rank && (
        <div className="absolute left-3 top-3 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-gradient-to-br from-yellow-400 to-orange-500 text-xs font-bold text-white">
          {rank}
        </div>
      )}

      {/* Waveform & Play Button */}
      <div className="relative h-32 bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
        <WaveformVisualization
          data={track.waveformData}
          isPlaying={track.isPlaying}
          height={96}
          className="h-full w-full"
          animateOnPlay
        />
        
        <motion.button
          onClick={handlePlay}
          className="absolute inset-0 flex items-center justify-center bg-black/20"
          initial={{ opacity: 0 }}
          animate={{ opacity: isHovered || track.isPlaying ? 1 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <motion.div
            className="flex h-12 w-12 items-center justify-center rounded-full bg-white/90 backdrop-blur-sm"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
          >
            {track.isPlaying ? (
              <PauseIcon className="h-6 w-6 text-gray-900" />
            ) : (
              <PlayIcon className="h-6 w-6 text-gray-900" />
            )}
          </motion.div>
        </motion.button>
      </div>

      {/* Content */}
      <div className="p-4">
        {/* Header */}
        <div className="mb-3 flex items-start justify-between">
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-lg font-medium text-gray-900">{track.title}</h3>
            <div className="mt-1 flex items-center space-x-2">
              <Avatar src={track.user.avatar} size="xs" />
              <span className="text-sm text-gray-600">{track.user.name}</span>
            </div>
          </div>

          <button
            onClick={() => setShowMenu(!showMenu)}
            className="text-gray-400 hover:text-gray-600"
          >
            <EllipsisHorizontalIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Description */}
        {track.description && (
          <p className="mb-3 line-clamp-2 text-sm text-gray-600">{track.description}</p>
        )}

        {/* Genre & Duration */}
        <div className="mb-3 flex items-center justify-between">
          <Badge variant="secondary">{track.genre}</Badge>
          <span className="text-sm text-gray-500">{formatTime(track.duration)}</span>
        </div>

        {/* Stats */}
        <div className="flex items-center justify-between text-sm text-gray-500">
          <span>{formatNumber(track.stats.plays)} plays</span>
          <span>{new Date(track.createdAt).toLocaleDateString()}</span>
        </div>

        {/* Actions */}
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <button
              onClick={handleLike}
              className="flex items-center space-x-1 text-gray-400 hover:text-red-500"
            >
              {track.isLiked ? (
                <HeartSolidIcon className="h-5 w-5 text-red-500" />
              ) : (
                <HeartIcon className="h-5 w-5" />
              )}
              <span>{formatNumber(track.stats.likes)}</span>
            </button>

            <button
              onClick={handleComment}
              className="flex items-center space-x-1 text-gray-400 hover:text-blue-500"
            >
              <ChatBubbleLeftIcon className="h-5 w-5" />
              <span>{formatNumber(track.stats.comments)}</span>
            </button>
          </div>

          <button
            onClick={handleShare}
            className="flex items-center space-x-1 text-gray-400 hover:text-green-500"
          >
            <ShareIcon className="h-5 w-5" />
            <span>{formatNumber(track.stats.shares)}</span>
          </button>
        </div>

        {/* Follow Button */}
        {showFollowButton && !track.user.isFollowing && (
          <div className="mt-3">
            <Button
              size="sm"
              variant="outline"
              onClick={handleFollow}
              className="w-full"
            >
              <UserPlusIcon className="mr-2 h-4 w-4" />
              Follow {track.user.name}
            </Button>
          </div>
        )}
      </div>
    </motion.div>
  )
}