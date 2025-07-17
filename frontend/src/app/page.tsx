import { Metadata } from 'next'
import { HeroSection } from '@/components/landing/HeroSection'
import { FeaturesSection } from '@/components/landing/FeaturesSection'
import { DemoSection } from '@/components/landing/DemoSection'
import { PricingSection } from '@/components/landing/PricingSection'
import { CommunitySection } from '@/components/landing/CommunitySection'
import { CTASection } from '@/components/landing/CTASection'

export const metadata: Metadata = {
  title: 'Professional AI Music Generation Platform',
  description: 'Create studio-quality music with advanced AI. Generate, edit, and share professional compositions with real-time collaboration.',
}

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <HeroSection />
      
      {/* Features Section */}
      <FeaturesSection />
      
      {/* Interactive Demo */}
      <DemoSection />
      
      {/* Community Showcase */}
      <CommunitySection />
      
      {/* Pricing */}
      <PricingSection />
      
      {/* Final CTA */}
      <CTASection />
    </div>
  )
}