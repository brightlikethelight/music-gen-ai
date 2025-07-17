# Interactive API Documentation

The Music Gen AI API provides interactive documentation through multiple interfaces, allowing you to explore endpoints, test requests, and understand responses in real-time.

## Available Documentation Interfaces

### 1. Swagger UI (Primary)
**URL**: [https://api.musicgen.ai/docs](https://api.musicgen.ai/docs)

Modern, interactive interface with:
- Live API testing
- Request/response examples
- Authentication integration
- Model schema visualization

### 2. Redoc (Alternative)
**URL**: [https://api.musicgen.ai/redoc](https://api.musicgen.ai/redoc)

Clean, three-panel documentation with:
- Detailed descriptions
- Code samples in multiple languages
- Responsive design
- Print-friendly layout

### 3. RapiDoc (Advanced)
**URL**: [https://api.musicgen.ai/rapidoc](https://api.musicgen.ai/rapidoc)

Modern documentation with:
- Dark/light themes
- Try-it-out functionality
- Schema explorer
- Multiple layout options

## Getting Started with Interactive Docs

### 1. Authentication Setup

Before testing endpoints, set up authentication:

**Option A: API Key Authentication**
1. Go to [https://api.musicgen.ai/docs](https://api.musicgen.ai/docs)
2. Click the "Authorize" button (🔓)
3. Enter your API key in the "ApiKeyAuth" section: `sk_live_your_key_here`
4. Click "Authorize"

**Option B: JWT Authentication**
1. Use the `/auth/login` endpoint to get a JWT token
2. Click "Authorize" in the docs
3. Enter `Bearer your_jwt_token_here` in the "JwtAuth" section
4. Click "Authorize"

### 2. Testing Your First Request

1. Navigate to the "Models" section
2. Expand `GET /models`
3. Click "Try it out"
4. Click "Execute"
5. View the response with available models

### 3. Generating Music

1. Go to the "Music Generation" section
2. Expand `POST /generate`
3. Click "Try it out"
4. Modify the request body:
```json
{
  "prompt": "Upbeat jazz with saxophone solo",
  "duration": 30,
  "model": "musicgen-small"
}
```
5. Click "Execute"
6. Copy the `task_id` from the response
7. Use the `GET /generate/{task_id}` endpoint to check status

## Local Development Setup

### Running Interactive Docs Locally

If you're running the API locally, you can access documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **Redoc**: http://localhost:8000/redoc

### Custom Documentation Server

You can also run a standalone documentation server:

```bash
# Install dependencies
npm install -g swagger-ui-dist redoc-cli

# Serve Swagger UI
cd docs/api
python -m http.server 8080

# Access at http://localhost:8080/swagger-ui.html
```

### Docker Setup

Run documentation in Docker:

```dockerfile
# Dockerfile.docs
FROM nginx:alpine

# Copy OpenAPI spec
COPY docs/api/openapi_spec.yaml /usr/share/nginx/html/

# Copy custom HTML files
COPY docs/api/swagger-ui.html /usr/share/nginx/html/
COPY docs/api/redoc.html /usr/share/nginx/html/

# Copy nginx config
COPY docs/api/nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
```

```bash
# Build and run
docker build -f Dockerfile.docs -t musicgen-docs .
docker run -p 8080:80 musicgen-docs
```

## Custom Documentation Pages

### Swagger UI HTML

Create a custom Swagger UI page:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Music Gen AI - API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        html {
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }
        
        *, *:before, *:after {
            box-sizing: inherit;
        }

        body {
            margin:0;
            background: #fafafa;
        }

        .swagger-ui .topbar {
            background-color: #1f1f1f;
        }

        .swagger-ui .topbar .download-url-wrapper .select-label {
            color: #ffffff;
        }

        .swagger-ui .info .title {
            color: #3b4151;
        }
    </style>
</head>

<body>
    <div id="swagger-ui"></div>

    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: './openapi_spec.yaml',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                requestInterceptor: function(request) {
                    // Add custom headers
                    request.headers['User-Agent'] = 'MusicGen-Docs/1.0';
                    return request;
                },
                responseInterceptor: function(response) {
                    // Log responses for debugging
                    console.log('API Response:', response);
                    return response;
                },
                onComplete: function(swaggerApi, swaggerUi) {
                    console.log("Loaded SwaggerUI");
                },
                persistAuthorization: true,
                tryItOutEnabled: true,
                filter: true,
                syntaxHighlight: {
                    activate: true,
                    theme: "tomorrow-night"
                }
            });

            window.ui = ui;
        };
    </script>
</body>
</html>
```

### Redoc HTML

Create a custom Redoc page:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Music Gen AI - API Reference</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 0;
        }
        
        redoc {
            display: block;
        }
    </style>
</head>
<body>
    <redoc spec-url='./openapi_spec.yaml'
           theme='{
               "colors": {
                   "primary": {
                       "main": "#1976d2"
                   }
               },
               "typography": {
                   "fontSize": "14px",
                   "lineHeight": "1.5em",
                   "code": {
                       "fontSize": "13px"
                   },
                   "headings": {
                       "fontFamily": "Montserrat, sans-serif",
                       "fontWeight": "400"
                   }
               },
               "sidebar": {
                   "backgroundColor": "#fafafa"
               }
           }'
           lazy-rendering
           native-scrollbars
           path-in-middle-panel
           suppress-warnings></redoc>
    
    <script src="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js"></script>
</body>
</html>
```

## Advanced Features

### Custom Code Samples

Add custom code samples to the OpenAPI spec:

```yaml
paths:
  /generate:
    post:
      # ... other properties
      x-codeSamples:
        - lang: 'Python'
          source: |
            import requests
            
            response = requests.post(
                'https://api.musicgen.ai/v1/generate',
                headers={
                    'X-API-Key': 'sk_live_your_key_here',
                    'Content-Type': 'application/json'
                },
                json={
                    'prompt': 'Upbeat jazz with saxophone solo',
                    'duration': 30,
                    'model': 'musicgen-small'
                }
            )
            
            print(response.json())
        
        - lang: 'JavaScript'
          source: |
            const response = await fetch('https://api.musicgen.ai/v1/generate', {
                method: 'POST',
                headers: {
                    'X-API-Key': 'sk_live_your_key_here',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: 'Upbeat jazz with saxophone solo',
                    duration: 30,
                    model: 'musicgen-small'
                })
            });
            
            const data = await response.json();
            console.log(data);
        
        - lang: 'cURL'
          source: |
            curl -X POST https://api.musicgen.ai/v1/generate \
                 -H "X-API-Key: sk_live_your_key_here" \
                 -H "Content-Type: application/json" \
                 -d '{
                     "prompt": "Upbeat jazz with saxophone solo",
                     "duration": 30,
                     "model": "musicgen-small"
                 }'
```

### Interactive Examples

Add interactive widgets to your documentation:

```html
<!-- Music Generation Widget -->
<div id="music-generator-widget">
    <h3>Try Music Generation</h3>
    <form id="generation-form">
        <div class="form-group">
            <label for="prompt">Music Prompt:</label>
            <input type="text" id="prompt" value="Upbeat jazz with saxophone solo" required>
        </div>
        
        <div class="form-group">
            <label for="duration">Duration (seconds):</label>
            <input type="number" id="duration" value="30" min="1" max="300" required>
        </div>
        
        <div class="form-group">
            <label for="model">Model:</label>
            <select id="model">
                <option value="musicgen-small">MusicGen Small (Fast)</option>
                <option value="musicgen-medium">MusicGen Medium</option>
                <option value="musicgen-large">MusicGen Large (Best)</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="api-key">Your API Key:</label>
            <input type="password" id="api-key" placeholder="sk_live_..." required>
        </div>
        
        <button type="submit">Generate Music</button>
    </form>
    
    <div id="generation-result" style="display: none;">
        <h4>Generation Result:</h4>
        <pre id="result-json"></pre>
        <div id="audio-player"></div>
    </div>
</div>

<script>
document.getElementById('generation-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = {
        prompt: document.getElementById('prompt').value,
        duration: parseInt(document.getElementById('duration').value),
        model: document.getElementById('model').value
    };
    
    const apiKey = document.getElementById('api-key').value;
    
    try {
        const response = await fetch('https://api.musicgen.ai/v1/generate', {
            method: 'POST',
            headers: {
                'X-API-Key': apiKey,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        document.getElementById('result-json').textContent = JSON.stringify(result, null, 2);
        document.getElementById('generation-result').style.display = 'block';
        
        if (response.ok) {
            // Poll for completion
            pollGenerationStatus(result.task_id, apiKey);
        }
    } catch (error) {
        document.getElementById('result-json').textContent = 'Error: ' + error.message;
        document.getElementById('generation-result').style.display = 'block';
    }
});

async function pollGenerationStatus(taskId, apiKey) {
    const maxAttempts = 30;
    let attempts = 0;
    
    const poll = async () => {
        try {
            const response = await fetch(`https://api.musicgen.ai/v1/generate/${taskId}`, {
                headers: { 'X-API-Key': apiKey }
            });
            
            const status = await response.json();
            document.getElementById('result-json').textContent = JSON.stringify(status, null, 2);
            
            if (status.status === 'completed' && status.audio_url) {
                // Create audio player
                const audioPlayer = document.getElementById('audio-player');
                audioPlayer.innerHTML = `
                    <audio controls>
                        <source src="${status.audio_url}" type="audio/wav">
                        Your browser does not support audio playback.
                    </audio>
                `;
            } else if (status.status === 'failed') {
                console.error('Generation failed:', status.error);
            } else if (status.status !== 'completed' && attempts < maxAttempts) {
                attempts++;
                setTimeout(poll, 2000); // Poll every 2 seconds
            }
        } catch (error) {
            console.error('Error polling status:', error);
        }
    };
    
    poll();
}
</script>

<style>
#music-generator-widget {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    background: #f9f9f9;
}

.form-group {
    margin-bottom: 15px;
}

.form-group label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

.form-group input,
.form-group select {
    width: 100%;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
}

button[type="submit"] {
    background: #1976d2;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

button[type="submit"]:hover {
    background: #1565c0;
}

#generation-result {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
}

#result-json {
    background: #f5f5f5;
    padding: 10px;
    border-radius: 4px;
    overflow-x: auto;
    white-space: pre-wrap;
}
</style>
```

## Documentation Hosting

### GitHub Pages

Host documentation on GitHub Pages:

1. Create a `docs/` directory in your repository
2. Add your HTML files and OpenAPI spec
3. Enable GitHub Pages in repository settings
4. Documentation will be available at `https://username.github.io/repository/`

### Netlify

Deploy documentation with Netlify:

```toml
# netlify.toml
[build]
  publish = "docs/api"
  command = "echo 'Static docs, no build needed'"

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-XSS-Protection = "1; mode=block"
    X-Content-Type-Options = "nosniff"

[[redirects]]
  from = "/"
  to = "/swagger-ui.html"
  status = 302
```

### CloudFlare Pages

Deploy with CloudFlare Pages:

```yaml
# .github/workflows/deploy-docs.yml
name: Deploy Documentation

on:
  push:
    branches: [ main ]
    paths: [ 'docs/**' ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Publish to Cloudflare Pages
        uses: cloudflare/pages-action@v1
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          projectName: musicgen-docs
          directory: docs/api
```

## Embedding Documentation

### WordPress/CMS Integration

Embed documentation in your website:

```html
<iframe 
    src="https://api.musicgen.ai/docs" 
    width="100%" 
    height="600px" 
    frameborder="0">
</iframe>
```

### React Component

Create a React component for your documentation:

```jsx
import React, { useEffect, useRef } from 'react';
import SwaggerUI from 'swagger-ui-react';
import 'swagger-ui-react/swagger-ui.css';

const APIDocumentation = ({ apiKey }) => {
    const requestInterceptor = (request) => {
        if (apiKey && !request.headers['X-API-Key']) {
            request.headers['X-API-Key'] = apiKey;
        }
        return request;
    };

    return (
        <div className="api-documentation">
            <SwaggerUI
                url="https://api.musicgen.ai/openapi.yaml"
                requestInterceptor={requestInterceptor}
                tryItOutEnabled={true}
                filter={true}
                layout="BaseLayout"
                deepLinking={true}
                displayOperationId={true}
                defaultModelsExpandDepth={1}
                defaultModelExpandDepth={1}
                showExtensions={true}
                showCommonExtensions={true}
            />
        </div>
    );
};

export default APIDocumentation;
```

## Analytics and Monitoring

### Documentation Usage Analytics

Track how users interact with your documentation:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'GA_MEASUREMENT_ID');

// Track API endpoint interactions
document.addEventListener('click', function(e) {
    if (e.target.matches('.opblock-summary-path')) {
        gtag('event', 'endpoint_clicked', {
            'custom_parameter': e.target.textContent
        });
    }
    
    if (e.target.matches('.btn.execute')) {
        gtag('event', 'try_it_out_executed', {
            'custom_parameter': 'api_test'
        });
    }
});
</script>

<!-- Hotjar for user behavior -->
<script>
(function(h,o,t,j,a,r){
    h.hj=h.hj||function(){(h.hj.q=h.hj.q||[]).push(arguments)};
    h._hjSettings={hjid:YOUR_HOTJAR_ID,hjsv:6};
    a=o.getElementsByTagName('head')[0];
    r=o.createElement('script');r.async=1;
    r.src=t+h._hjSettings.hjid+j+h._hjSettings.hjsv;
    a.appendChild(r);
})(window,document,'https://static.hotjar.com/c/hotjar-','.js?sv=');
</script>
```

### Documentation Performance Monitoring

Monitor documentation performance:

```javascript
// Performance monitoring
window.addEventListener('load', function() {
    const perfData = performance.getEntriesByType('navigation')[0];
    
    // Send performance data to analytics
    gtag('event', 'page_load_time', {
        'value': Math.round(perfData.loadEventEnd - perfData.loadEventStart),
        'custom_parameter': 'docs_performance'
    });
});

// Monitor API test results
const originalFetch = window.fetch;
window.fetch = function(...args) {
    const [url] = args;
    
    return originalFetch.apply(this, arguments)
        .then(response => {
            if (url.includes('musicgen.ai')) {
                gtag('event', 'api_test_request', {
                    'custom_parameter': response.status >= 200 && response.status < 300 ? 'success' : 'error'
                });
            }
            return response;
        })
        .catch(error => {
            if (url.includes('musicgen.ai')) {
                gtag('event', 'api_test_request', {
                    'custom_parameter': 'network_error'
                });
            }
            throw error;
        });
};
```

## Maintenance and Updates

### Automated Documentation Updates

Keep documentation in sync with API changes:

```yaml
# .github/workflows/update-docs.yml
name: Update API Documentation

on:
  push:
    branches: [ main ]
    paths: [ 'music_gen/api/**' ]

jobs:
  update-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Generate OpenAPI Spec
        run: |
          python -m music_gen.api.generate_openapi_spec > docs/api/openapi_spec.yaml
      
      - name: Validate OpenAPI Spec
        run: |
          npx swagger-cli validate docs/api/openapi_spec.yaml
      
      - name: Deploy Documentation
        run: |
          # Deploy to your hosting platform
          # e.g., Netlify, Vercel, GitHub Pages
          
      - name: Notify Team
        run: |
          # Send notification about documentation updates
          curl -X POST $SLACK_WEBHOOK_URL \
               -H 'Content-Type: application/json' \
               -d '{"text":"API documentation updated automatically"}'
```

### Version Management

Maintain documentation for multiple API versions:

```
docs/
├── api/
│   ├── v1/
│   │   ├── openapi_spec.yaml
│   │   ├── swagger-ui.html
│   │   └── redoc.html
│   ├── v2/
│   │   ├── openapi_spec.yaml
│   │   ├── swagger-ui.html
│   │   └── redoc.html
│   └── latest -> v2/
```

## Support and Feedback

For documentation-related issues:

1. **Report Issues**: [GitHub Issues](https://github.com/musicgen-ai/docs/issues)
2. **Request Features**: [Feature Requests](https://github.com/musicgen-ai/docs/discussions)
3. **Documentation Support**: [docs@musicgen.ai](mailto:docs@musicgen.ai)
4. **Community**: [Discord](https://discord.gg/musicgen) #documentation channel

### Contributing to Documentation

We welcome contributions to improve our documentation:

1. Fork the documentation repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request
5. Our team will review and merge approved changes

See our [Documentation Contributing Guide](https://github.com/musicgen-ai/docs/blob/main/CONTRIBUTING.md) for detailed instructions.