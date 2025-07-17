'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  ChartBarIcon,
  UsersIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  EyeIcon,
  PlayIcon,
  PauseIcon,
} from '@heroicons/react/24/outline'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { testScenarios, TestScenario } from '@/lib/testScenarios'
import { useUserTesting } from '@/hooks/useUserTesting'

interface TestingMetrics {
  totalSessions: number
  completionRate: number
  averageTime: number
  errorRate: number
  satisfactionScore: number
  dropoffPoints: Array<{ step: string; percentage: number }>
}

interface UserFeedback {
  id: string
  userId: string
  scenario: string
  rating: number
  comment: string
  timestamp: number
  device: string
}

export function TestingDashboard() {
  const [selectedScenario, setSelectedScenario] = useState<TestScenario | null>(null)
  const [isRunningTest, setIsRunningTest] = useState(false)
  const [metrics, setMetrics] = useState<TestingMetrics | null>(null)
  const [recentFeedback, setRecentFeedback] = useState<UserFeedback[]>([])
  const [timeRange, setTimeRange] = useState<'1d' | '7d' | '30d'>('7d')
  
  const {
    currentScenario,
    testResults,
    startTestScenario,
    completeTestScenario,
    isTracking,
  } = useUserTesting()

  // Mock data - in real implementation, fetch from API
  useEffect(() => {
    setMetrics({
      totalSessions: 1247,
      completionRate: 78.5,
      averageTime: 245,
      errorRate: 12.3,
      satisfactionScore: 4.2,
      dropoffPoints: [
        { step: 'Navigation to Studio', percentage: 5.2 },
        { step: 'Parameter Adjustment', percentage: 15.8 },
        { step: 'Audio Generation', percentage: 8.9 },
        { step: 'Playback Testing', percentage: 3.1 },
      ],
    })

    setRecentFeedback([
      {
        id: '1',
        userId: 'user123',
        scenario: 'First-Time Music Generation',
        rating: 5,
        comment: 'Really intuitive interface! Loved how easy it was to generate music.',
        timestamp: Date.now() - 1000 * 60 * 30,
        device: 'Desktop',
      },
      {
        id: '2',
        userId: 'user456',
        scenario: 'Mobile Generation Flow',
        rating: 3,
        comment: 'Mobile interface could be improved. Some buttons are too small.',
        timestamp: Date.now() - 1000 * 60 * 60 * 2,
        device: 'Mobile',
      },
      {
        id: '3',
        userId: 'user789',
        scenario: 'Audio Editing Workflow',
        rating: 4,
        comment: 'Editor is powerful but has a learning curve. Could use better tutorials.',
        timestamp: Date.now() - 1000 * 60 * 60 * 5,
        device: 'Desktop',
      },
    ])
  }, [timeRange])

  const startTest = (scenario: TestScenario) => {
    setSelectedScenario(scenario)
    setIsRunningTest(true)
    startTestScenario(scenario, 'test-user-' + Date.now())
  }

  const stopTest = async () => {
    if (currentScenario) {
      await completeTestScenario('Test completed from dashboard', 5)
    }
    setIsRunningTest(false)
    setSelectedScenario(null)
  }

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}m ${remainingSeconds}s`
  }

  const getRatingColor = (rating: number) => {
    if (rating >= 4) return 'text-green-400'
    if (rating >= 3) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white">User Testing Dashboard</h1>
          <p className="text-gray-300 mt-2">
            Monitor user testing scenarios and gather insights to improve the experience
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <Select value={timeRange} onValueChange={(value: any) => setTimeRange(value)}>
            <SelectTrigger className="w-32 bg-gray-800 border-gray-700 text-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1d">Last 24h</SelectItem>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Active Test Status */}
      {isRunningTest && currentScenario && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-purple-900 border border-purple-600 rounded-lg p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">
                🧪 Active Test: {currentScenario.name}
              </h3>
              <p className="text-purple-200">
                Status: {isTracking ? 'Recording user actions...' : 'Paused'}
              </p>
              {testResults && (
                <div className="mt-2 text-sm text-purple-200">
                  Tasks completed: {testResults.tasksCompleted}/{testResults.totalTasks} •
                  Errors: {testResults.errors} •
                  Time: {formatTime(Math.floor((Date.now() - testResults.startTime) / 1000))}
                </div>
              )}
            </div>
            
            <Button
              onClick={stopTest}
              variant="outline"
              className="border-purple-400 text-purple-400 hover:bg-purple-400 hover:text-white"
            >
              Stop Test
            </Button>
          </div>
        </motion.div>
      )}

      {/* Metrics Overview */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-300 flex items-center">
                <UsersIcon className="h-4 w-4 mr-2" />
                Total Sessions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{metrics.totalSessions.toLocaleString()}</div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-300 flex items-center">
                <CheckCircleIcon className="h-4 w-4 mr-2" />
                Completion Rate
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-400">{metrics.completionRate}%</div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-300 flex items-center">
                <ClockIcon className="h-4 w-4 mr-2" />
                Avg. Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-400">{formatTime(metrics.averageTime)}</div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-300 flex items-center">
                <XCircleIcon className="h-4 w-4 mr-2" />
                Error Rate
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-400">{metrics.errorRate}%</div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-300 flex items-center">
                <ChartBarIcon className="h-4 w-4 mr-2" />
                Satisfaction
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-400">{metrics.satisfactionScore}/5</div>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Test Scenarios */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">Test Scenarios</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {testScenarios.map((scenario) => (
                <div key={scenario.id} className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
                  <div className="flex-1">
                    <h4 className="font-medium text-white">{scenario.name}</h4>
                    <p className="text-sm text-gray-300 mt-1">{scenario.description}</p>
                    <div className="flex items-center space-x-4 mt-2 text-xs text-gray-400">
                      <span>Duration: {formatTime(scenario.estimatedDuration)}</span>
                      <span>Tasks: {scenario.tasks.length}</span>
                      <span className={`px-2 py-1 rounded ${
                        scenario.difficulty === 'beginner' ? 'bg-green-900 text-green-200' :
                        scenario.difficulty === 'intermediate' ? 'bg-yellow-900 text-yellow-200' :
                        'bg-red-900 text-red-200'
                      }`}>
                        {scenario.difficulty}
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-gray-600 text-gray-300 hover:bg-gray-600"
                    >
                      <EyeIcon className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      onClick={() => startTest(scenario)}
                      disabled={isRunningTest}
                      className="bg-purple-600 hover:bg-purple-700"
                    >
                      {isRunningTest ? <PauseIcon className="h-4 w-4" /> : <PlayIcon className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recent Feedback */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">Recent User Feedback</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentFeedback.map((feedback) => (
                <div key={feedback.id} className="p-4 bg-gray-700 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <div className={`text-lg font-bold ${getRatingColor(feedback.rating)}`}>
                        {feedback.rating}/5
                      </div>
                      <span className="text-gray-400">•</span>
                      <span className="text-sm text-gray-400">{feedback.scenario}</span>
                    </div>
                    <div className="text-xs text-gray-500">
                      {new Date(feedback.timestamp).toLocaleDateString()}
                    </div>
                  </div>
                  
                  <p className="text-gray-300 text-sm mb-2">{feedback.comment}</p>
                  
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>User: {feedback.userId}</span>
                    <span>{feedback.device}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Drop-off Analysis */}
      {metrics && (
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">User Drop-off Points</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {metrics.dropoffPoints.map((point, index) => (
                <div key={index} className="flex items-center justify-between">
                  <span className="text-gray-300">{point.step}</span>
                  <div className="flex items-center space-x-3">
                    <div className="w-32 bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-red-500 h-2 rounded-full"
                        style={{ width: `${point.percentage}%` }}
                      />
                    </div>
                    <span className="text-red-400 text-sm font-medium w-12 text-right">
                      {point.percentage}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}