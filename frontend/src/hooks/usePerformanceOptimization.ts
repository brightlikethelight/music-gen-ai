/**
 * Performance optimization hook for mobile and slow connections.
 * 
 * Implements 2024 best practices for mobile web performance including
 * lazy loading, bundle optimization, and adaptive loading strategies.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

interface PerformanceMetrics {
  connectionType: string;
  effectiveType: string;
  downlink: number;
  rtt: number;
  saveData: boolean;
}

interface OptimizationConfig {
  enableLazyLoading: boolean;
  enableImageOptimization: boolean;
  enableAdaptiveLoading: boolean;
  maxConcurrentRequests: number;
  requestTimeout: number;
  enablePrefetching: boolean;
}

interface LoadingStrategy {
  shouldLoadImages: boolean;
  shouldLoadVideo: boolean;
  shouldPreloadRoutes: boolean;
  shouldUseLowQuality: boolean;
  imageQuality: 'low' | 'medium' | 'high';
  audioQuality: 'low' | 'medium' | 'high';
}

/**
 * Hook for comprehensive performance optimization
 */
export const usePerformanceOptimization = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [config, setConfig] = useState<OptimizationConfig>({
    enableLazyLoading: true,
    enableImageOptimization: true,
    enableAdaptiveLoading: true,
    maxConcurrentRequests: 3,
    requestTimeout: 10000,
    enablePrefetching: false
  });
  const [loadingStrategy, setLoadingStrategy] = useState<LoadingStrategy>({
    shouldLoadImages: true,
    shouldLoadVideo: true,
    shouldPreloadRoutes: true,
    shouldUseLowQuality: false,
    imageQuality: 'high',
    audioQuality: 'high'
  });

  // Initialize performance monitoring
  useEffect(() => {
    const updateMetrics = () => {
      if ('connection' in navigator) {
        const connection = (navigator as any).connection;
        setMetrics({
          connectionType: connection.type || 'unknown',
          effectiveType: connection.effectiveType || '4g',
          downlink: connection.downlink || 10,
          rtt: connection.rtt || 0,
          saveData: connection.saveData || false
        });
      }
    };

    // Initial metrics
    updateMetrics();

    // Listen for connection changes
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      connection.addEventListener('change', updateMetrics);
      
      return () => {
        connection.removeEventListener('change', updateMetrics);
      };
    }
  }, []);

  // Adaptive loading strategy based on connection
  useEffect(() => {
    if (!metrics) return;

    const isSlowConnection = metrics.effectiveType === 'slow-2g' || 
                           metrics.effectiveType === '2g' ||
                           metrics.downlink < 1.5;
    
    const isSaveDataEnabled = metrics.saveData;

    const newStrategy: LoadingStrategy = {
      shouldLoadImages: !isSlowConnection,
      shouldLoadVideo: !isSlowConnection && !isSaveDataEnabled,
      shouldPreloadRoutes: !isSlowConnection && !isSaveDataEnabled,
      shouldUseLowQuality: isSlowConnection || isSaveDataEnabled,
      imageQuality: isSlowConnection ? 'low' : isSaveDataEnabled ? 'medium' : 'high',
      audioQuality: isSlowConnection ? 'low' : isSaveDataEnabled ? 'medium' : 'high'
    };

    setLoadingStrategy(newStrategy);

    // Update configuration based on connection
    setConfig(prev => ({
      ...prev,
      maxConcurrentRequests: isSlowConnection ? 1 : 3,
      requestTimeout: isSlowConnection ? 15000 : 10000,
      enablePrefetching: !isSlowConnection && !isSaveDataEnabled
    }));
  }, [metrics]);

  return {
    metrics,
    config,
    loadingStrategy,
    isSlowConnection: metrics?.effectiveType === 'slow-2g' || metrics?.effectiveType === '2g',
    isSaveDataEnabled: metrics?.saveData || false
  };
};

/**
 * Hook for lazy loading with intersection observer
 */
export const useLazyLoading = (options: IntersectionObserverInit = {}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);
  const elementRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element || hasLoaded) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          setHasLoaded(true);
          observer.unobserve(element);
        }
      },
      {
        threshold: 0.1,
        rootMargin: '50px',
        ...options
      }
    );

    observer.observe(element);

    return () => {
      observer.unobserve(element);
    };
  }, [hasLoaded, options]);

  return { elementRef, isVisible, hasLoaded };
};

/**
 * Hook for adaptive image loading
 */
export const useAdaptiveImages = () => {
  const { loadingStrategy } = usePerformanceOptimization();

  const getOptimizedImageUrl = useCallback((
    baseUrl: string,
    options: {
      width?: number;
      height?: number;
      quality?: 'low' | 'medium' | 'high';
      format?: 'webp' | 'jpg' | 'png';
    } = {}
  ) => {
    if (!loadingStrategy.shouldLoadImages) {
      return ''; // Return empty string for placeholder
    }

    const quality = options.quality || loadingStrategy.imageQuality;
    const format = options.format || 'webp';
    
    // Quality mapping
    const qualityMap = {
      low: 30,
      medium: 60,
      high: 85
    };

    const params = new URLSearchParams({
      ...(options.width && { w: options.width.toString() }),
      ...(options.height && { h: options.height.toString() }),
      q: qualityMap[quality].toString(),
      f: format
    });

    return `${baseUrl}?${params.toString()}`;
  }, [loadingStrategy]);

  return { getOptimizedImageUrl, shouldLoadImages: loadingStrategy.shouldLoadImages };
};

/**
 * Hook for request queue management
 */
export const useRequestQueue = () => {
  const { config } = usePerformanceOptimization();
  const [queue, setQueue] = useState<Array<() => Promise<any>>>([]);
  const [activeRequests, setActiveRequests] = useState(0);

  const addToQueue = useCallback((request: () => Promise<any>) => {
    setQueue(prev => [...prev, request]);
  }, []);

  const processQueue = useCallback(async () => {
    if (activeRequests >= config.maxConcurrentRequests || queue.length === 0) {
      return;
    }

    const nextRequest = queue[0];
    setQueue(prev => prev.slice(1));
    setActiveRequests(prev => prev + 1);

    try {
      await nextRequest();
    } catch (error) {
      console.error('Request failed:', error);
    } finally {
      setActiveRequests(prev => prev - 1);
    }
  }, [queue, activeRequests, config.maxConcurrentRequests]);

  useEffect(() => {
    processQueue();
  }, [queue, activeRequests, processQueue]);

  return { addToQueue, queueLength: queue.length, activeRequests };
};

/**
 * Hook for preloading critical routes
 */
export const useRoutePreloading = () => {
  const { loadingStrategy } = usePerformanceOptimization();
  const preloadedRoutes = useRef<Set<string>>(new Set());

  const preloadRoute = useCallback(async (routePath: string) => {
    if (!loadingStrategy.shouldPreloadRoutes || preloadedRoutes.current.has(routePath)) {
      return;
    }

    try {
      // Mark as preloaded immediately to prevent duplicates
      preloadedRoutes.current.add(routePath);

      // Preload the route component
      const module = await import(/* webpackChunkName: "route-[request]" */ `../pages${routePath}`);
      
      console.log(`Preloaded route: ${routePath}`);
    } catch (error) {
      console.warn(`Failed to preload route ${routePath}:`, error);
      // Remove from preloaded set on failure
      preloadedRoutes.current.delete(routePath);
    }
  }, [loadingStrategy.shouldPreloadRoutes]);

  const preloadCriticalRoutes = useCallback(() => {
    const criticalRoutes = ['/studio', '/community', '/profile'];
    criticalRoutes.forEach(route => preloadRoute(route));
  }, [preloadRoute]);

  return { preloadRoute, preloadCriticalRoutes };
};

/**
 * Hook for audio quality adaptation
 */
export const useAdaptiveAudio = () => {
  const { loadingStrategy } = usePerformanceOptimization();

  const getAudioUrl = useCallback((baseUrl: string, options: {
    format?: 'mp3' | 'ogg' | 'wav';
    bitrate?: number;
  } = {}) => {
    const qualityBitrates = {
      low: 64,
      medium: 128,
      high: 320
    };

    const bitrate = options.bitrate || qualityBitrates[loadingStrategy.audioQuality];
    const format = options.format || 'mp3';

    const params = new URLSearchParams({
      bitrate: bitrate.toString(),
      format
    });

    return `${baseUrl}?${params.toString()}`;
  }, [loadingStrategy.audioQuality]);

  return { 
    getAudioUrl, 
    audioQuality: loadingStrategy.audioQuality,
    shouldUseLowQuality: loadingStrategy.shouldUseLowQuality
  };
};

/**
 * Hook for bundle size monitoring
 */
export const useBundleOptimization = () => {
  const [bundleStats, setBundleStats] = useState<{
    loadTime: number;
    scriptCount: number;
    totalSize: number;
  } | null>(null);

  useEffect(() => {
    // Monitor page load performance
    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      
      const scriptEntries = entries.filter(entry => 
        entry.name.includes('.js') || entry.name.includes('.chunk')
      );

      if (scriptEntries.length > 0) {
        const totalSize = scriptEntries.reduce((sum, entry) => {
          return sum + (entry as any).transferSize || 0;
        }, 0);

        setBundleStats({
          loadTime: performance.now(),
          scriptCount: scriptEntries.length,
          totalSize
        });
      }
    });

    observer.observe({ entryTypes: ['resource'] });

    return () => observer.disconnect();
  }, []);

  return bundleStats;
};

/**
 * Utility for critical resource hints
 */
export const addResourceHints = () => {
  const { config } = usePerformanceOptimization();

  useEffect(() => {
    if (!config.enablePrefetching) return;

    // Add DNS prefetch for external domains
    const dnsPrefetchDomains = [
      'https://api.musicgenai.com',
      'https://cdn.musicgenai.com'
    ];

    dnsPrefetchDomains.forEach(domain => {
      const link = document.createElement('link');
      link.rel = 'dns-prefetch';
      link.href = domain;
      document.head.appendChild(link);
    });

    // Preconnect to critical domains
    const preconnectDomains = ['https://api.musicgenai.com'];
    
    preconnectDomains.forEach(domain => {
      const link = document.createElement('link');
      link.rel = 'preconnect';
      link.href = domain;
      link.crossOrigin = 'anonymous';
      document.head.appendChild(link);
    });
  }, [config.enablePrefetching]);
};