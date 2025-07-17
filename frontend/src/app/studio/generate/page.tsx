import { Metadata } from 'next'
import { GenerationStudio } from '@/components/generation/GenerationStudio'

export const metadata: Metadata = {
  title: 'Music Generation Studio',
  description: 'Create professional music with AI. Generate, customize, and export high-quality compositions.',
}

export default function GeneratePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <GenerationStudio />
    </div>
  )
}