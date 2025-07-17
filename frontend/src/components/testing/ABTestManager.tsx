'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  BeakerIcon,
  PlayIcon,
  PauseIcon,
  StopIcon,
  ChartBarIcon,
  UserGroupIcon,
  CalendarIcon,
  PlusIcon,
  EyeIcon,
  PencilIcon,
  TrashIcon,
} from '@heroicons/react/24/outline'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Input } from '@/components/ui/Input'
import { Textarea } from '@/components/ui/Textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { ABTest, ABVariant } from '@/hooks/useABTesting'

interface ABTestManagerProps {
  onTestCreate?: (test: ABTest) => void
  onTestUpdate?: (test: ABTest) => void
  onTestDelete?: (testId: string) => void
}

export function ABTestManager({ onTestCreate, onTestUpdate, onTestDelete }: ABTestManagerProps) {
  const [tests, setTests] = useState<ABTest[]>([])
  const [isCreating, setIsCreating] = useState(false)
  const [editingTest, setEditingTest] = useState<ABTest | null>(null)
  const [newTest, setNewTest] = useState<Partial<ABTest>>({
    name: '',
    description: '',
    hypothesis: '',
    targetMetric: '',
    trafficAllocation: 10,
    variants: [
      {
        id: 'control',
        name: 'Control',
        description: 'Original version',
        weight: 50,
        config: {},
        isControl: true,
      },
      {
        id: 'variant-a',
        name: 'Variant A',
        description: 'Test version',
        weight: 50,
        config: {},
        isControl: false,
      },
    ],
  })

  // Mock data - in real implementation, fetch from API
  useEffect(() => {
    setTests([
      {
        id: 'generation-button-color',
        name: 'Generation Button Color Test',
        description: 'Testing purple vs blue generate button for higher conversion',
        status: 'running',
        startDate: '2024-01-15',
        endDate: '2024-02-15',
        trafficAllocation: 50,
        targetMetric: 'generation_clicks',
        hypothesis: 'Blue button will increase click-through rate by 15%',
        variants: [
          {
            id: 'control',
            name: 'Purple Button (Control)',
            description: 'Current purple generate button',
            weight: 50,
            config: { buttonColor: 'purple' },
            isControl: true,
          },
          {
            id: 'blue-button',
            name: 'Blue Button',
            description: 'Blue generate button variant',
            weight: 50,
            config: { buttonColor: 'blue' },
            isControl: false,
          },
        ],
      },
      {
        id: 'onboarding-flow',
        name: 'Simplified Onboarding',
        description: 'Testing simplified vs detailed onboarding flow',
        status: 'draft',
        startDate: '2024-02-01',
        trafficAllocation: 25,
        targetMetric: 'onboarding_completion',
        hypothesis: 'Simplified onboarding will improve completion rate by 20%',
        variants: [
          {
            id: 'control',
            name: 'Current Onboarding',
            description: 'Existing detailed onboarding',
            weight: 50,
            config: { onboardingType: 'detailed' },
            isControl: true,
          },
          {
            id: 'simplified',
            name: 'Simplified Flow',
            description: 'Streamlined onboarding process',
            weight: 50,
            config: { onboardingType: 'simplified' },
            isControl: false,
          },
        ],
      },
      {
        id: 'mobile-player-layout',
        name: 'Mobile Player Layout',
        description: 'Testing bottom vs floating audio player on mobile',
        status: 'paused',
        startDate: '2024-01-10',
        trafficAllocation: 30,
        targetMetric: 'mobile_engagement',
        hypothesis: 'Floating player will increase mobile engagement by 10%',
        segments: ['mobile_users'],
        variants: [
          {
            id: 'control',
            name: 'Bottom Player',
            description: 'Fixed bottom audio player',
            weight: 50,
            config: { playerPosition: 'bottom' },
            isControl: true,
          },
          {
            id: 'floating',
            name: 'Floating Player',
            description: 'Floating audio player overlay',
            weight: 50,
            config: { playerPosition: 'floating' },
            isControl: false,
          },
        ],
      },
    ])
  }, [])

  const getStatusColor = (status: ABTest['status']) => {
    switch (status) {
      case 'running': return 'text-green-400 bg-green-900'
      case 'paused': return 'text-yellow-400 bg-yellow-900'
      case 'completed': return 'text-blue-400 bg-blue-900'
      case 'draft': return 'text-gray-400 bg-gray-700'
      default: return 'text-gray-400 bg-gray-700'
    }
  }

  const getStatusIcon = (status: ABTest['status']) => {
    switch (status) {
      case 'running': return PlayIcon
      case 'paused': return PauseIcon
      case 'completed': return StopIcon
      case 'draft': return PencilIcon
      default: return PencilIcon
    }
  }

  const updateTestStatus = async (testId: string, newStatus: ABTest['status']) => {
    setTests(prev => prev.map(test => 
      test.id === testId ? { ...test, status: newStatus } : test
    ))
    
    // In real implementation, call API
    console.log(`Updated test ${testId} status to ${newStatus}`)
  }

  const deleteTest = async (testId: string) => {
    if (window.confirm('Are you sure you want to delete this test?')) {
      setTests(prev => prev.filter(test => test.id !== testId))
      onTestDelete?.(testId)
    }
  }

  const addVariant = () => {
    const variantId = `variant-${String.fromCharCode(65 + (newTest.variants?.length || 0))}`
    const newVariant: ABVariant = {
      id: variantId,
      name: `Variant ${variantId.slice(-1)}`,
      description: '',
      weight: 50,
      config: {},
      isControl: false,
    }
    
    setNewTest(prev => ({
      ...prev,
      variants: [...(prev.variants || []), newVariant],
    }))
  }

  const removeVariant = (variantId: string) => {
    setNewTest(prev => ({
      ...prev,
      variants: prev.variants?.filter(v => v.id !== variantId),
    }))
  }

  const updateVariant = (variantId: string, updates: Partial<ABVariant>) => {
    setNewTest(prev => ({
      ...prev,
      variants: prev.variants?.map(v => 
        v.id === variantId ? { ...v, ...updates } : v
      ),
    }))
  }

  const createTest = async () => {
    if (!newTest.name || !newTest.description || !newTest.targetMetric) {
      alert('Please fill in all required fields')
      return
    }

    const test: ABTest = {
      id: `test-${Date.now()}`,
      name: newTest.name!,
      description: newTest.description!,
      hypothesis: newTest.hypothesis || '',
      targetMetric: newTest.targetMetric!,
      trafficAllocation: newTest.trafficAllocation || 10,
      status: 'draft',
      startDate: new Date().toISOString().split('T')[0],
      variants: newTest.variants || [],
    }

    setTests(prev => [...prev, test])
    onTestCreate?.(test)
    setIsCreating(false)
    
    // Reset form
    setNewTest({
      name: '',
      description: '',
      hypothesis: '',
      targetMetric: '',
      trafficAllocation: 10,
      variants: [
        {
          id: 'control',
          name: 'Control',
          description: 'Original version',
          weight: 50,
          config: {},
          isControl: true,
        },
        {
          id: 'variant-a',
          name: 'Variant A',
          description: 'Test version',
          weight: 50,
          config: {},
          isControl: false,
        },
      ],
    })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center">
            <BeakerIcon className="h-8 w-8 mr-3 text-purple-400" />
            A/B Test Manager
          </h2>
          <p className="text-gray-300 mt-2">
            Create and manage A/B tests to optimize user experience
          </p>
        </div>
        
        <Button
          onClick={() => setIsCreating(true)}
          className="bg-purple-600 hover:bg-purple-700 flex items-center space-x-2"
        >
          <PlusIcon className="h-5 w-5" />
          <span>Create Test</span>
        </Button>
      </div>

      {/* Create Test Form */}
      {isCreating && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800 rounded-lg border border-gray-700 p-6"
        >
          <h3 className="text-lg font-semibold text-white mb-4">Create New A/B Test</h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Test Name *
                </label>
                <Input
                  value={newTest.name || ''}
                  onChange={(e) => setNewTest(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="e.g., Button Color Test"
                  className="bg-gray-700 border-gray-600 text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Description *
                </label>
                <Textarea
                  value={newTest.description || ''}
                  onChange={(e) => setNewTest(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Describe what you're testing"
                  className="bg-gray-700 border-gray-600 text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Hypothesis
                </label>
                <Textarea
                  value={newTest.hypothesis || ''}
                  onChange={(e) => setNewTest(prev => ({ ...prev, hypothesis: e.target.value }))}
                  placeholder="What do you expect to happen?"
                  className="bg-gray-700 border-gray-600 text-white"
                />
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Target Metric *
                </label>
                <Select
                  value={newTest.targetMetric || ''}
                  onValueChange={(value) => setNewTest(prev => ({ ...prev, targetMetric: value }))}
                >
                  <SelectTrigger className="bg-gray-700 border-gray-600 text-white">
                    <SelectValue placeholder="Select metric to track" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="generation_clicks">Generation Clicks</SelectItem>
                    <SelectItem value="onboarding_completion">Onboarding Completion</SelectItem>
                    <SelectItem value="mobile_engagement">Mobile Engagement</SelectItem>
                    <SelectItem value="user_retention">User Retention</SelectItem>
                    <SelectItem value="conversion_rate">Conversion Rate</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Traffic Allocation (%)
                </label>
                <Input
                  type="number"
                  min="1"
                  max="100"
                  value={newTest.trafficAllocation || 10}
                  onChange={(e) => setNewTest(prev => ({ ...prev, trafficAllocation: parseInt(e.target.value) }))}
                  className="bg-gray-700 border-gray-600 text-white"
                />
              </div>

              {/* Variants */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <label className="block text-sm font-medium text-gray-300">
                    Test Variants
                  </label>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={addVariant}
                    className="border-gray-600 text-gray-300 hover:bg-gray-700"
                  >
                    <PlusIcon className="h-4 w-4 mr-1" />
                    Add Variant
                  </Button>
                </div>
                
                <div className="space-y-3">
                  {newTest.variants?.map((variant, index) => (
                    <div key={variant.id} className="bg-gray-700 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <Input
                          value={variant.name}
                          onChange={(e) => updateVariant(variant.id, { name: e.target.value })}
                          className="bg-gray-600 border-gray-500 text-white text-sm"
                          placeholder="Variant name"
                        />
                        
                        {!variant.isControl && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => removeVariant(variant.id)}
                            className="text-red-400 hover:text-red-300 ml-2"
                          >
                            <TrashIcon className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2">
                        <Input
                          value={variant.description}
                          onChange={(e) => updateVariant(variant.id, { description: e.target.value })}
                          className="bg-gray-600 border-gray-500 text-white text-sm"
                          placeholder="Description"
                        />
                        <Input
                          type="number"
                          value={variant.weight}
                          onChange={(e) => updateVariant(variant.id, { weight: parseInt(e.target.value) })}
                          className="bg-gray-600 border-gray-500 text-white text-sm"
                          placeholder="Weight %"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="flex justify-end space-x-3 mt-6">
            <Button
              variant="ghost"
              onClick={() => setIsCreating(false)}
              className="text-gray-400 hover:text-white"
            >
              Cancel
            </Button>
            <Button
              onClick={createTest}
              className="bg-purple-600 hover:bg-purple-700"
            >
              Create Test
            </Button>
          </div>
        </motion.div>
      )}

      {/* Test List */}
      <div className="grid grid-cols-1 gap-6">
        {tests.map((test) => {
          const StatusIcon = getStatusIcon(test.status)
          
          return (
            <Card key={test.id} className="bg-gray-800 border-gray-700">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <CardTitle className="text-white text-lg">{test.name}</CardTitle>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(test.status)}`}>
                      <StatusIcon className="h-3 w-3 inline mr-1" />
                      {test.status}
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-gray-600 text-gray-300 hover:bg-gray-700"
                    >
                      <ChartBarIcon className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-gray-600 text-gray-300 hover:bg-gray-700"
                    >
                      <EyeIcon className="h-4 w-4" />
                    </Button>
                    {test.status === 'running' ? (
                      <Button
                        size="sm"
                        onClick={() => updateTestStatus(test.id, 'paused')}
                        className="bg-yellow-600 hover:bg-yellow-700"
                      >
                        <PauseIcon className="h-4 w-4" />
                      </Button>
                    ) : test.status === 'paused' ? (
                      <Button
                        size="sm"
                        onClick={() => updateTestStatus(test.id, 'running')}
                        className="bg-green-600 hover:bg-green-700"
                      >
                        <PlayIcon className="h-4 w-4" />
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        onClick={() => updateTestStatus(test.id, 'running')}
                        className="bg-purple-600 hover:bg-purple-700"
                      >
                        <PlayIcon className="h-4 w-4" />
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => deleteTest(test.id)}
                      className="text-red-400 hover:text-red-300"
                    >
                      <TrashIcon className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent>
                <div className="space-y-4">
                  <p className="text-gray-300">{test.description}</p>
                  
                  {test.hypothesis && (
                    <div className="bg-gray-700 rounded-lg p-3">
                      <h4 className="text-sm font-medium text-gray-300 mb-1">Hypothesis:</h4>
                      <p className="text-sm text-gray-400">{test.hypothesis}</p>
                    </div>
                  )}
                  
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                    <div>
                      <div className="flex items-center text-gray-400 mb-1">
                        <CalendarIcon className="h-4 w-4 mr-1" />
                        Start Date
                      </div>
                      <div className="text-white">{test.startDate}</div>
                    </div>
                    
                    <div>
                      <div className="flex items-center text-gray-400 mb-1">
                        <UserGroupIcon className="h-4 w-4 mr-1" />
                        Traffic
                      </div>
                      <div className="text-white">{test.trafficAllocation}%</div>
                    </div>
                    
                    <div>
                      <div className="flex items-center text-gray-400 mb-1">
                        <ChartBarIcon className="h-4 w-4 mr-1" />
                        Target Metric
                      </div>
                      <div className="text-white">{test.targetMetric}</div>
                    </div>
                    
                    <div>
                      <div className="text-gray-400 mb-1">Variants</div>
                      <div className="text-white">{test.variants.length}</div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    {test.variants.map((variant) => (
                      <div key={variant.id} className="flex items-center justify-between bg-gray-700 rounded-lg p-3">
                        <div>
                          <div className="font-medium text-white">
                            {variant.name}
                            {variant.isControl && (
                              <span className="ml-2 px-2 py-1 bg-blue-900 text-blue-200 text-xs rounded">
                                Control
                              </span>
                            )}
                          </div>
                          <div className="text-sm text-gray-400">{variant.description}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-white font-medium">{variant.weight}%</div>
                          <div className="text-xs text-gray-400">allocation</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}