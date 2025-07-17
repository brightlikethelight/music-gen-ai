/**
 * Mobile-optimized Webpack configuration for Music Gen AI.
 * 
 * Implements 2024 best practices for mobile bundle optimization,
 * code splitting, and performance optimization.
 */

const path = require('path');
const webpack = require('webpack');
const CompressionPlugin = require('compression-webpack-plugin');
const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
const TerserPlugin = require('terser-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const OptimizeCSSAssetsPlugin = require('optimize-css-assets-webpack-plugin');
const WorkboxPlugin = require('workbox-webpack-plugin');

const isProduction = process.env.NODE_ENV === 'production';
const shouldAnalyze = process.env.ANALYZE === 'true';

module.exports = {
  mode: isProduction ? 'production' : 'development',
  
  entry: {
    // Main application entry
    main: './src/index.tsx',
    
    // Vendor chunk for stable caching
    vendor: [
      'react',
      'react-dom',
      'react-router-dom'
    ]
  },

  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: isProduction 
      ? 'static/js/[name].[contenthash:8].js'
      : 'static/js/[name].js',
    chunkFilename: isProduction
      ? 'static/js/[name].[contenthash:8].chunk.js'
      : 'static/js/[name].chunk.js',
    publicPath: '/',
    clean: true
  },

  resolve: {
    extensions: ['.tsx', '.ts', '.js', '.jsx'],
    alias: {
      '@': path.resolve(__dirname, 'src'),
      '@components': path.resolve(__dirname, 'src/components'),
      '@hooks': path.resolve(__dirname, 'src/hooks'),
      '@utils': path.resolve(__dirname, 'src/utils'),
      '@types': path.resolve(__dirname, 'src/types')
    }
  },

  module: {
    rules: [
      // TypeScript/JavaScript
      {
        test: /\.(ts|tsx|js|jsx)$/,
        exclude: /node_modules/,
        use: [
          {
            loader: 'babel-loader',
            options: {
              presets: [
                ['@babel/preset-env', {
                  targets: {
                    browsers: ['> 1%', 'last 2 versions', 'not ie <= 11']
                  },
                  modules: false,
                  useBuiltIns: 'usage',
                  corejs: 3
                }],
                '@babel/preset-react',
                '@babel/preset-typescript'
              ],
              plugins: [
                // Dynamic imports for code splitting
                '@babel/plugin-syntax-dynamic-import',
                
                // React optimizations
                ['@babel/plugin-transform-react-jsx', {
                  runtime: 'automatic'
                }],
                
                // Remove unused code in production
                ...(isProduction ? [
                  ['babel-plugin-transform-react-remove-prop-types', {
                    removeImport: true
                  }]
                ] : [])
              ],
              cacheDirectory: true,
              cacheCompression: false
            }
          }
        ]
      },

      // CSS/SCSS
      {
        test: /\.(css|scss|sass)$/,
        use: [
          isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
          {
            loader: 'css-loader',
            options: {
              modules: {
                auto: true,
                localIdentName: isProduction 
                  ? '[hash:base64:8]'
                  : '[name]__[local]__[hash:base64:5]'
              },
              importLoaders: 2
            }
          },
          {
            loader: 'postcss-loader',
            options: {
              postcssOptions: {
                plugins: [
                  'autoprefixer',
                  'postcss-preset-env',
                  ...(isProduction ? ['cssnano'] : [])
                ]
              }
            }
          },
          'sass-loader'
        ]
      },

      // Images with optimization
      {
        test: /\.(png|jpe?g|gif|webp|svg)$/i,
        type: 'asset',
        parser: {
          dataUrlCondition: {
            maxSize: 8192 // 8KB
          }
        },
        generator: {
          filename: 'static/media/[name].[hash:8][ext]'
        },
        use: [
          {
            loader: 'image-webpack-loader',
            options: {
              mozjpeg: {
                progressive: true,
                quality: 80
              },
              optipng: {
                enabled: true
              },
              pngquant: {
                quality: [0.65, 0.80],
                speed: 4
              },
              gifsicle: {
                interlaced: false
              },
              webp: {
                quality: 80,
                method: 6
              }
            }
          }
        ]
      },

      // Audio files
      {
        test: /\.(mp3|wav|ogg|m4a)$/,
        type: 'asset/resource',
        generator: {
          filename: 'static/media/[name].[hash:8][ext]'
        }
      },

      // Fonts
      {
        test: /\.(woff|woff2|eot|ttf|otf)$/,
        type: 'asset/resource',
        generator: {
          filename: 'static/fonts/[name].[hash:8][ext]'
        }
      }
    ]
  },

  optimization: {
    minimize: isProduction,
    minimizer: [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: isProduction,
            drop_debugger: isProduction,
            pure_funcs: isProduction ? ['console.log', 'console.info'] : []
          },
          mangle: {
            safari10: true
          },
          format: {
            comments: false
          }
        },
        extractComments: false
      }),
      new OptimizeCSSAssetsPlugin({
        cssProcessorOptions: {
          map: {
            inline: false,
            annotation: true
          }
        }
      })
    ],

    // Advanced code splitting strategy
    splitChunks: {
      chunks: 'all',
      minSize: 20000,
      maxSize: 244000,
      cacheGroups: {
        // Vendor libraries
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          priority: 10,
          chunks: 'all'
        },
        
        // React ecosystem
        react: {
          test: /[\\/]node_modules[\\/](react|react-dom|react-router)[\\/]/,
          name: 'react',
          priority: 20,
          chunks: 'all'
        },
        
        // UI libraries
        ui: {
          test: /[\\/]node_modules[\\/](@mui|@emotion|styled-components)[\\/]/,
          name: 'ui',
          priority: 15,
          chunks: 'all'
        },
        
        // Audio processing libraries
        audio: {
          test: /[\\/]node_modules[\\/](wavesurfer|tone|web-audio)[\\/]/,
          name: 'audio',
          priority: 15,
          chunks: 'all'
        },
        
        // Common chunks
        common: {
          name: 'common',
          minChunks: 2,
          priority: 5,
          chunks: 'all',
          enforce: true
        }
      }
    },

    // Runtime chunk for better caching
    runtimeChunk: {
      name: 'runtime'
    },

    // Module concatenation for better compression
    concatenateModules: isProduction,

    // Tree shaking
    usedExports: true,
    sideEffects: false
  },

  plugins: [
    // Environment variables
    new webpack.DefinePlugin({
      'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV),
      'process.env.REACT_APP_API_URL': JSON.stringify(process.env.REACT_APP_API_URL || 'http://localhost:8000'),
      __DEV__: !isProduction
    }),

    // Extract CSS in production
    ...(isProduction ? [
      new MiniCssExtractPlugin({
        filename: 'static/css/[name].[contenthash:8].css',
        chunkFilename: 'static/css/[name].[contenthash:8].chunk.css'
      })
    ] : []),

    // Compression
    ...(isProduction ? [
      new CompressionPlugin({
        algorithm: 'gzip',
        test: /\.(js|css|html|svg)$/,
        threshold: 8192,
        minRatio: 0.8
      })
    ] : []),

    // Progressive Web App
    ...(isProduction ? [
      new WorkboxPlugin.GenerateSW({
        clientsClaim: true,
        skipWaiting: true,
        maximumFileSizeToCacheInBytes: 5 * 1024 * 1024, // 5MB
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\.musicgenai\.com\//,
            handler: 'StaleWhileRevalidate',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 300 // 5 minutes
              }
            }
          },
          {
            urlPattern: /\.(png|jpg|jpeg|svg|gif|webp)$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'images-cache',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 86400 // 24 hours
              }
            }
          },
          {
            urlPattern: /\.(mp3|wav|ogg)$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'audio-cache',
              expiration: {
                maxEntries: 50,
                maxAgeSeconds: 86400 // 24 hours
              }
            }
          }
        ]
      })
    ] : []),

    // Bundle analyzer
    ...(shouldAnalyze ? [
      new BundleAnalyzerPlugin({
        analyzerMode: 'server',
        openAnalyzer: true
      })
    ] : []),

    // Progress plugin for build feedback
    new webpack.ProgressPlugin()
  ],

  // Development server configuration
  devServer: {
    port: 3000,
    hot: true,
    historyApiFallback: true,
    compress: true,
    client: {
      overlay: {
        errors: true,
        warnings: false
      }
    }
  },

  // Performance hints for mobile optimization
  performance: {
    hints: isProduction ? 'warning' : false,
    maxEntrypointSize: 250000, // 250KB
    maxAssetSize: 250000, // 250KB
    assetFilter: (assetFilename) => {
      // Only check JavaScript and CSS files
      return /\.(js|css)$/.test(assetFilename);
    }
  },

  // Source maps for debugging
  devtool: isProduction ? 'source-map' : 'eval-cheap-module-source-map',

  // Cache configuration for faster builds
  cache: {
    type: 'filesystem',
    buildDependencies: {
      config: [__filename]
    }
  },

  // Experiments for latest features
  experiments: {
    topLevelAwait: true
  }
};

// Mobile-specific optimizations
if (process.env.MOBILE_BUILD === 'true') {
  module.exports.resolve.alias['react-dom'] = 'react-dom/profiling';
  
  // Reduce bundle size for mobile
  module.exports.optimization.splitChunks.maxSize = 200000; // 200KB max chunks
  module.exports.performance.maxEntrypointSize = 200000; // 200KB
  module.exports.performance.maxAssetSize = 200000; // 200KB
  
  // Additional mobile plugins
  module.exports.plugins.push(
    new webpack.NormalModuleReplacementPlugin(
      /^lodash$/,
      'lodash-es'
    )
  );
}