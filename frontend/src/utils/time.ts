/**
 * Format time in seconds to MM:SS format
 * @param seconds - Time in seconds
 * @returns Formatted time string (MM:SS)
 */
export function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.floor(seconds % 60)
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
}

/**
 * Format time in seconds to HH:MM:SS format
 * @param seconds - Time in seconds
 * @returns Formatted time string (HH:MM:SS)
 */
export function formatLongTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainingSeconds = Math.floor(seconds % 60)
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
  }
  
  return formatTime(seconds)
}

/**
 * Parse time string (MM:SS or HH:MM:SS) to seconds
 * @param timeString - Time string to parse
 * @returns Time in seconds
 */
export function parseTime(timeString: string): number {
  const parts = timeString.split(':').map(Number)
  
  if (parts.length === 2) {
    // MM:SS format
    return parts[0] * 60 + parts[1]
  } else if (parts.length === 3) {
    // HH:MM:SS format
    return parts[0] * 3600 + parts[1] * 60 + parts[2]
  }
  
  return 0
}

/**
 * Format duration in a human-readable way
 * @param seconds - Duration in seconds
 * @returns Human-readable duration string
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = Math.floor(seconds % 60)
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`
  } else {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`
  }
}

/**
 * Convert BPM and time signature to beat duration
 * @param bpm - Beats per minute
 * @param timeSignature - Time signature (e.g., "4/4")
 * @returns Beat duration in seconds
 */
export function getBeatDuration(bpm: number, timeSignature: string = '4/4'): number {
  const [numerator, denominator] = timeSignature.split('/').map(Number)
  const beatValue = 4 / denominator // Quarter note = 1, eighth note = 0.5, etc.
  return (60 / bpm) * beatValue
}

/**
 * Convert seconds to beat position
 * @param seconds - Time in seconds
 * @param bpm - Beats per minute
 * @param timeSignature - Time signature
 * @returns Beat position
 */
export function secondsToBeats(seconds: number, bpm: number, timeSignature: string = '4/4'): number {
  const beatDuration = getBeatDuration(bpm, timeSignature)
  return seconds / beatDuration
}

/**
 * Convert beat position to seconds
 * @param beats - Beat position
 * @param bpm - Beats per minute
 * @param timeSignature - Time signature
 * @returns Time in seconds
 */
export function beatsToSeconds(beats: number, bpm: number, timeSignature: string = '4/4'): number {
  const beatDuration = getBeatDuration(bpm, timeSignature)
  return beats * beatDuration
}