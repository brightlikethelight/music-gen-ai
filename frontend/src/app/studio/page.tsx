import { Metadata } from 'next'
import { GenerationStudio } from '@/components/generation/GenerationStudio'
import { ResponsiveContainer, ResponsiveSection } from '@/components/layout/ResponsiveLayout'

export const metadata: Metadata = {
  title: 'Music Studio',
  description: 'Create AI-generated music with our advanced studio interface.',
}

export default function StudioPage() {
  return (
    <ResponsiveSection spacing="lg">
      <ResponsiveContainer size="xl">
        <div className="space-y-8">
          {/* Header */}
          <div className="text-center lg:text-left">
            <h1 className="text-3xl lg:text-4xl font-bold text-white mb-4">
              Music Generation Studio
            </h1>
            <p className="text-lg text-gray-300 max-w-2xl mx-auto lg:mx-0">
              Transform your ideas into professional-quality music using advanced AI technology.
              Describe your vision and let our AI compose the perfect soundtrack.
            </p>
          </div>

          {/* Studio Interface */}
          <div className="bg-gray-800 rounded-2xl border border-gray-700 overflow-hidden">
            <GenerationStudio />
          </div>

          {/* Quick Tips - Mobile Optimized */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-3">
                🎵 Describe Your Vision
              </h3>
              <p className="text-gray-300 text-sm">
                Use natural language to describe the mood, genre, and style you want.
                Be as specific or creative as you'd like!
              </p>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-3">
                ⚡ Real-time Generation
              </h3>
              <p className="text-gray-300 text-sm">
                Watch your music come to life with real-time progress updates
                and preview capabilities during generation.
              </p>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 md:col-span-2 lg:col-span-1">
              <h3 className="text-lg font-semibold text-white mb-3">
                🎛️ Professional Tools
              </h3>
              <p className="text-gray-300 text-sm">
                Access advanced editing tools, effects, and export options
                for professional music production.
              </p>
            </div>
          </div>
        </div>
      </ResponsiveContainer>
    </ResponsiveSection>
  )
}