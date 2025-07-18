#!/usr/bin/env python3
"""
Test script to verify PyTorch-only transformers installation for MusicGen.
This script tests that transformers works without TensorFlow dependencies.
"""

import sys
import traceback
import warnings

def test_pytorch_import():
    """Test PyTorch import and basic functionality."""
    print("🔥 Testing PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.mm(x, x)
        print(f"✓ Basic tensor operations work")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        traceback.print_exc()
        return False

def test_transformers_import():
    """Test transformers import without TensorFlow."""
    print("\n🤗 Testing Transformers...")
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        
        # Test pipeline import
        from transformers import pipeline
        print("✓ Pipeline import successful")
        
        # Test AutoTokenizer and AutoModel
        from transformers import AutoTokenizer, AutoModel
        print("✓ AutoTokenizer and AutoModel imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Transformers test failed: {e}")
        traceback.print_exc()
        return False

def test_no_tensorflow():
    """Verify TensorFlow is not imported by transformers."""
    print("\n🚫 Testing TensorFlow absence...")
    try:
        # This should fail if TensorFlow is not installed
        import tensorflow
        print("⚠️  TensorFlow is still installed")
        return False
    except ImportError:
        print("✓ TensorFlow is not installed (good!)")
        return True
    except Exception as e:
        print(f"✓ TensorFlow import failed as expected: {e}")
        return True

def test_musicgen_dependencies():
    """Test MusicGen-related dependencies."""
    print("\n🎼 Testing MusicGen dependencies...")
    success = True
    
    dependencies = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("soundfile", "SoundFile"),
        ("librosa", "Librosa"),
        ("pydub", "PyDub"),
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
            success = False
    
    # Test audiocraft if available
    try:
        import audiocraft
        print("✓ AudioCraft imported successfully")
    except ImportError:
        print("⚠️  AudioCraft not available (install with: pip install audiocraft)")
    
    return success

def test_transformers_functionality():
    """Test basic transformers functionality."""
    print("\n🧪 Testing Transformers functionality...")
    try:
        from transformers import AutoTokenizer
        
        # Test with a small model (this might download ~500MB on first run)
        model_name = "distilbert-base-uncased"
        print(f"Testing with {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test tokenization
        text = "Hello, this is a test for MusicGen setup!"
        tokens = tokenizer(text, return_tensors="pt")
        print(f"✓ Tokenization successful: {tokens['input_ids'].shape}")
        
        print("✓ Basic transformers functionality works")
        return True
        
    except Exception as e:
        print(f"✗ Transformers functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_pytorch_transformers_integration():
    """Test PyTorch + Transformers integration."""
    print("\n🔗 Testing PyTorch + Transformers integration...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Test inference
        text = "Testing PyTorch integration"
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Model inference successful: {outputs.last_hidden_state.shape}")
        return True
        
    except Exception as e:
        print(f"✗ PyTorch + Transformers integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🎵 MusicGen PyTorch-only Setup Test")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Suppress some warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    tests = [
        ("PyTorch Import", test_pytorch_import),
        ("Transformers Import", test_transformers_import),
        ("TensorFlow Absence", test_no_tensorflow),
        ("MusicGen Dependencies", test_musicgen_dependencies),
        ("Transformers Functionality", test_transformers_functionality),
        ("PyTorch + Transformers Integration", test_pytorch_transformers_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your PyTorch-only transformers setup is working correctly.")
        print("\nYou can now use MusicGen with transformers without TensorFlow dependencies.")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please check the error messages above.")
        print("\nSuggested fixes:")
        print("1. Ensure PyTorch is installed: pip install torch>=2.2.0")
        print("2. Install transformers with PyTorch extras: pip install 'transformers[torch]>=4.31.0'")
        print("3. Remove TensorFlow: pip uninstall tensorflow tensorflow-datasets tensorflow_estimator tensorflow-metadata")
        print("4. Install missing dependencies from requirements-pytorch-only.txt")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)