import { Metadata } from 'next'
import { AudioEditor } from '@/components/editor/AudioEditor'
import { ResponsiveLayout } from '@/components/layout/ResponsiveLayout'

export const metadata: Metadata = {
  title: 'Audio Editor',
  description: 'Professional audio editing tools for your AI-generated music.',
}

export default function EditorPage() {
  return (
    <div className="h-screen">
      <AudioEditor />
    </div>
  )
}