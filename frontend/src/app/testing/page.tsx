import { Metadata } from 'next'
import { TestingDashboard } from '@/components/testing/TestingDashboard'
import { ABTestManager } from '@/components/testing/ABTestManager'
import { ResponsiveContainer, ResponsiveSection } from '@/components/layout/ResponsiveLayout'
import { useState } from 'react'

export const metadata: Metadata = {
  title: 'User Testing Dashboard',
  description: 'Monitor user testing scenarios and gather insights to improve the user experience.',
}

export default function TestingPage() {
  return (
    <ResponsiveSection spacing="lg">
      <ResponsiveContainer size="xl">
        <div className="space-y-12">
          {/* User Testing Dashboard */}
          <TestingDashboard />
          
          {/* A/B Test Manager */}
          <ABTestManager />
        </div>
      </ResponsiveContainer>
    </ResponsiveSection>
  )
}