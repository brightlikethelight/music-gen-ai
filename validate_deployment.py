#!/usr/bin/env python3
"""
MusicGen Deployment Validation Script

Educational validation script for testing deployment examples.
⚠️ ACADEMIC PROJECT - FOR LEARNING PURPOSES ONLY
"""

import os
import sys
import json
import time
import subprocess
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class DeploymentValidator:
    """Validates MusicGen deployment readiness."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "environment": {},
            "dependencies": {},
            "api_tests": {},
            "deployment_options": {},
            "recommendations": []
        }
        
    def print_header(self, text: str):
        """Print formatted header."""
        print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
        print(f"{BOLD}{BLUE}{text.center(60)}{RESET}")
        print(f"{BOLD}{BLUE}{'=' * 60}{RESET}")
        
    def print_test(self, name: str, passed: bool, details: str = ""):
        """Print test result."""
        status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
        print(f"  {name:<40} {status}")
        if details:
            print(f"    {YELLOW}{details}{RESET}")
            
    def check_python_version(self) -> Tuple[bool, str]:
        """Check Python version compatibility."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        self.results["environment"]["python_version"] = version_str
        
        if version.major == 3 and version.minor == 12:
            return False, f"Python {version_str} is NOT compatible with ML ecosystem"
        elif version.major == 3 and version.minor in [10, 11]:
            return True, f"Python {version_str} is compatible"
        else:
            return False, f"Python {version_str} is untested"
            
    def check_dependencies(self) -> Dict[str, bool]:
        """Check required dependencies."""
        deps = {
            "torch": None,
            "transformers": None,
            "fastapi": None,
            "numpy": None,
            "scipy": None,
            "audiocraft": None
        }
        
        for dep in deps:
            try:
                module = __import__(dep)
                version = getattr(module, "__version__", "unknown")
                deps[dep] = (True, version)
            except ImportError:
                deps[dep] = (False, "not installed")
                
        self.results["dependencies"] = {k: v[1] for k, v in deps.items()}
        return deps
        
    def check_gpu(self) -> Tuple[bool, str]:
        """Check GPU availability."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                details = f"{gpu_name} ({memory_gb:.1f}GB)"
                self.results["environment"]["gpu"] = details
                return True, details
            else:
                self.results["environment"]["gpu"] = "Not available"
                return False, "No CUDA GPU available"
        except:
            self.results["environment"]["gpu"] = "Error checking"
            return False, "Could not check GPU"
            
    def check_docker(self) -> Tuple[bool, str]:
        """Check Docker availability."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                self.results["environment"]["docker"] = version
                return True, version
            else:
                self.results["environment"]["docker"] = "Not available"
                return False, "Docker not installed"
        except:
            self.results["environment"]["docker"] = "Not available"
            return False, "Docker not available"
            
    def test_api_endpoint(self, url: str, timeout: int = 5) -> Tuple[bool, str]:
        """Test API endpoint availability."""
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True, f"Status {response.status_code}"
            else:
                return False, f"Status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Connection refused"
        except requests.exceptions.Timeout:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
            
    def test_local_api(self) -> Dict[str, bool]:
        """Test local API if running."""
        base_url = "http://localhost:8000"
        endpoints = {
            "health": f"{base_url}/health",
            "models": f"{base_url}/models",
            "docs": f"{base_url}/docs"
        }
        
        results = {}
        for name, url in endpoints.items():
            passed, details = self.test_api_endpoint(url)
            results[name] = (passed, details)
            
        self.results["api_tests"] = {k: v[1] for k, v in results.items()}
        return results
        
    def check_deployment_options(self) -> Dict[str, Tuple[bool, str]]:
        """Check available deployment options."""
        options = {}
        
        # 1. Docker deployment
        docker_ok, docker_msg = self.check_docker()
        if docker_ok:
            # Check for our images
            try:
                result = subprocess.run(
                    ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                    capture_output=True,
                    text=True
                )
                images = result.stdout.strip().split('\n')
                if any('musicgen' in img for img in images):
                    options["docker_custom"] = (True, "Custom image available")
                else:
                    options["docker_custom"] = (False, "No custom image built")
                    
                if any('ashleykza/tts-webui' in img for img in images):
                    options["docker_prebuilt"] = (True, "Pre-built image available")
                else:
                    options["docker_prebuilt"] = (False, "Pre-built image not pulled")
            except:
                options["docker_custom"] = (False, "Cannot check images")
                options["docker_prebuilt"] = (False, "Cannot check images")
        else:
            options["docker_custom"] = (False, "Docker not available")
            options["docker_prebuilt"] = (False, "Docker not available")
            
        # 2. Local Python environment
        py_ok, py_msg = self.check_python_version()
        if py_ok:
            deps = self.check_dependencies()
            critical_deps = ["torch", "transformers", "fastapi"]
            if all(deps[d][0] for d in critical_deps):
                options["local_python"] = (True, "Critical dependencies installed")
            else:
                missing = [d for d in critical_deps if not deps[d][0]]
                options["local_python"] = (False, f"Missing: {', '.join(missing)}")
        else:
            options["local_python"] = (False, py_msg)
            
        # 3. Cloud deployment readiness
        options["cloud_replicate"] = (True, "Ready - requires API key")
        options["cloud_huggingface"] = (True, "Ready - requires account")
        options["cloud_runpod"] = (docker_ok, "Ready" if docker_ok else "Requires Docker")
        
        self.results["deployment_options"] = {k: v[1] for k, v in options.items()}
        return options
        
    def generate_recommendations(self):
        """Generate deployment recommendations."""
        recs = []
        
        # Python version
        py_version = self.results["environment"].get("python_version", "")
        if "3.12" in py_version:
            recs.append({
                "priority": "CRITICAL",
                "issue": "Python 3.12 is incompatible with ML dependencies",
                "solution": "Use Python 3.10 or 3.11 via pyenv or Docker"
            })
            
        # GPU
        if self.results["environment"].get("gpu") == "Not available":
            recs.append({
                "priority": "HIGH",
                "issue": "No GPU available",
                "solution": "Use cloud GPU services or expect slow CPU generation"
            })
            
        # Dependencies
        missing_deps = [k for k, v in self.results["dependencies"].items() 
                       if v == "not installed" and k != "audiocraft"]
        if missing_deps:
            recs.append({
                "priority": "HIGH",
                "issue": f"Missing dependencies: {', '.join(missing_deps)}",
                "solution": "Install with: pip install " + " ".join(missing_deps)
            })
            
        # Docker
        if self.results["environment"].get("docker") == "Not available":
            recs.append({
                "priority": "MEDIUM",
                "issue": "Docker not available",
                "solution": "Install Docker Desktop for easier deployment"
            })
            
        self.results["recommendations"] = recs
        
    def run_full_validation(self):
        """Run complete validation suite."""
        self.print_header("MusicGen Deployment Validation")
        
        # 1. Environment checks
        print(f"\n{BOLD}1. Environment Checks{RESET}")
        py_ok, py_msg = self.check_python_version()
        self.print_test("Python Version", py_ok, py_msg)
        
        gpu_ok, gpu_msg = self.check_gpu()
        self.print_test("GPU Availability", gpu_ok, gpu_msg)
        
        docker_ok, docker_msg = self.check_docker()
        self.print_test("Docker", docker_ok, docker_msg)
        
        # 2. Dependencies
        print(f"\n{BOLD}2. Python Dependencies{RESET}")
        deps = self.check_dependencies()
        for dep, (installed, version) in deps.items():
            self.print_test(dep, installed, version)
            
        # 3. API Tests
        print(f"\n{BOLD}3. API Tests (if running){RESET}")
        api_tests = self.test_local_api()
        if any(test[0] for test in api_tests.values()):
            for endpoint, (passed, details) in api_tests.items():
                self.print_test(f"GET /{endpoint}", passed, details)
        else:
            print(f"  {YELLOW}API not running locally{RESET}")
            
        # 4. Deployment Options
        print(f"\n{BOLD}4. Deployment Options{RESET}")
        options = self.check_deployment_options()
        for option, (available, details) in options.items():
            self.print_test(option.replace('_', ' ').title(), available, details)
            
        # 5. Recommendations
        self.generate_recommendations()
        if self.results["recommendations"]:
            print(f"\n{BOLD}5. Recommendations{RESET}")
            for rec in self.results["recommendations"]:
                priority_color = RED if rec["priority"] == "CRITICAL" else YELLOW
                print(f"\n  {priority_color}{rec['priority']}{RESET}: {rec['issue']}")
                print(f"  {GREEN}Solution{RESET}: {rec['solution']}")
                
        # 6. Summary
        self.print_summary()
        
        # Save report
        self.save_report()
        
    def print_summary(self):
        """Print validation summary."""
        self.print_header("Summary")
        
        # Determine overall status
        critical_issues = [r for r in self.results["recommendations"] 
                          if r["priority"] == "CRITICAL"]
        
        if critical_issues:
            print(f"\n{RED}{BOLD}❌ DEPLOYMENT BLOCKED{RESET}")
            print(f"{RED}Critical issues must be resolved:{RESET}")
            for issue in critical_issues:
                print(f"  - {issue['issue']}")
        else:
            print(f"\n{GREEN}{BOLD}✅ READY FOR DEPLOYMENT{RESET}")
            
            # Recommend best option
            options = self.results["deployment_options"]
            if options.get("docker_prebuilt", "").startswith("Pre-built"):
                print(f"\n{BOLD}Recommended:{RESET} Use pre-built Docker image")
                print("  docker run -d --gpus all -p 3000:3001 ashleykza/tts-webui:latest")
            elif options.get("local_python", "").startswith("Critical"):
                print(f"\n{BOLD}Recommended:{RESET} Run locally")
                print("  python -m musicgen.api.rest.app")
            else:
                print(f"\n{BOLD}Recommended:{RESET} Use cloud service")
                print("  - Replicate: https://replicate.com/meta/musicgen")
                print("  - Hugging Face: https://huggingface.co/spaces/facebook/MusicGen")
                
    def save_report(self):
        """Save validation report to file."""
        filename = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n{BLUE}Report saved to {filename}{RESET}")


def main():
    """Run validation."""
    validator = DeploymentValidator()
    
    # Add command line options
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # Quick check only
            py_ok, _ = validator.check_python_version()
            docker_ok, _ = validator.check_docker()
            print(f"Python OK: {py_ok}")
            print(f"Docker OK: {docker_ok}")
        elif sys.argv[1] == "--api":
            # Test API only
            api_tests = validator.test_local_api()
            for endpoint, (passed, details) in api_tests.items():
                print(f"{endpoint}: {'✓' if passed else '✗'} {details}")
    else:
        # Full validation
        validator.run_full_validation()


if __name__ == "__main__":
    main()