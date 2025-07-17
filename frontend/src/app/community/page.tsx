import { Metadata } from 'next'
import { CommunityHub } from '@/components/community/CommunityHub'
import { ResponsiveContainer, ResponsiveSection } from '@/components/layout/ResponsiveLayout'

export const metadata: Metadata = {
  title: 'Community Hub',
  description: 'Discover, share, and collaborate on AI-generated music with creators worldwide.',
}

export default function CommunityPage() {
  return (
    <ResponsiveSection spacing="lg">
      <ResponsiveContainer size="xl">
        <div className="space-y-8">
          {/* Community Header */}
          <div className="text-center lg:text-left">
            <h1 className="text-3xl lg:text-4xl font-bold text-white mb-4">
              Music Community
            </h1>
            <p className="text-lg text-gray-300 max-w-2xl mx-auto lg:mx-0">
              Connect with fellow creators, discover amazing music, and share your AI-generated compositions
              with a passionate community of music enthusiasts.
            </p>
          </div>

          {/* Community Stats - Responsive Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
            <div className="bg-gray-800 rounded-lg p-4 lg:p-6 border border-gray-700 text-center">
              <div className="text-2xl lg:text-3xl font-bold text-purple-400 mb-1">
                12.5K
              </div>
              <div className="text-sm lg:text-base text-gray-300">
                Active Creators
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-4 lg:p-6 border border-gray-700 text-center">
              <div className="text-2xl lg:text-3xl font-bold text-blue-400 mb-1">
                89.2K
              </div>
              <div className="text-sm lg:text-base text-gray-300">
                Tracks Shared
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-4 lg:p-6 border border-gray-700 text-center">
              <div className="text-2xl lg:text-3xl font-bold text-green-400 mb-1">
                456K
              </div>
              <div className="text-sm lg:text-base text-gray-300">
                Total Plays
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-4 lg:p-6 border border-gray-700 text-center">
              <div className="text-2xl lg:text-3xl font-bold text-yellow-400 mb-1">
                23.1K
              </div>
              <div className="text-sm lg:text-base text-gray-300">
                Collaborations
              </div>
            </div>
          </div>

          {/* Community Hub - Full Width on Mobile */}
          <div className="bg-gray-800 rounded-2xl border border-gray-700 overflow-hidden">
            <CommunityHub />
          </div>
        </div>
      </ResponsiveContainer>
    </ResponsiveSection>
  )
}