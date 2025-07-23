#!/usr/bin/env python3
"""
Integration script for V2 improvements
Run this to integrate the improved V2 implementations into your project
"""

import os
import shutil
import sys


def backup_original():
    """Backup original V2 file"""
    src = "pytorch_bfo_optimizer/optimizer_v2.py"
    dst = "pytorch_bfo_optimizer/optimizer_v2_original.py"
    
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"✓ Backed up original to {dst}")
        return True
    elif os.path.exists(dst):
        print(f"ℹ Backup already exists at {dst}")
        return True
    else:
        print(f"⚠ Original file not found at {src}")
        return False


def integrate_improvements():
    """Copy improved V2 to main location"""
    src = "pytorch_bfo_optimizer/optimizer_v2_improved.py"
    dst = "pytorch_bfo_optimizer/optimizer_v2.py"
    
    if not os.path.exists(src):
        print(f"✗ Improved file not found at {src}")
        return False
    
    shutil.copy2(src, dst)
    print(f"✓ Integrated improvements to {dst}")
    return True


def update_init_file():
    """Update __init__.py to include V2 exports"""
    init_file = "pytorch_bfo_optimizer/__init__.py"
    
    if not os.path.exists(init_file):
        print(f"⚠ __init__.py not found at {init_file}")
        return False
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Check if V2 imports already exist
    if "BFOv2" in content:
        print("ℹ V2 exports already in __init__.py")
        return True
    
    # Add V2 imports
    v2_imports = """
# V2 implementations (improved)
try:
    from .optimizer_v2 import BFOv2, AdaptiveBFOv2, HybridBFOv2
    __all__.extend(['BFOv2', 'AdaptiveBFOv2', 'HybridBFOv2'])
except ImportError:
    pass  # V2 not available
"""
    
    # Find where to insert (after original imports)
    if "__all__" in content:
        # Insert before __all__ extension
        parts = content.split("__all__")
        new_content = parts[0] + v2_imports + "\n__all__" + "__all__".join(parts[1:])
    else:
        # Append at the end
        new_content = content + "\n" + v2_imports
    
    with open(init_file, 'w') as f:
        f.write(new_content)
    
    print("✓ Updated __init__.py with V2 exports")
    return True


def run_tests():
    """Run improvement tests"""
    print("\nRunning tests...")
    
    test_file = "test_v2_improvements.py"
    if os.path.exists(test_file):
        os.system(f"{sys.executable} {test_file}")
    else:
        print(f"⚠ Test file not found: {test_file}")


def main():
    """Main integration process"""
    print("PyTorch BFO V2 Improvements Integration")
    print("=" * 50)
    
    steps = [
        ("Backing up original V2", backup_original),
        ("Integrating improvements", integrate_improvements),
        ("Updating package exports", update_init_file),
    ]
    
    success = True
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            success = False
            break
    
    if success:
        print("\n✅ Integration complete!")
        print("\nYou can now use the improved V2 implementations:")
        print("  from pytorch_bfo_optimizer import BFOv2, AdaptiveBFOv2, HybridBFOv2")
        
        response = input("\nRun tests? (y/n): ")
        if response.lower() == 'y':
            run_tests()
    else:
        print("\n✗ Integration failed. Please check the errors above.")
        
    print("\nRefer to MIGRATION_GUIDE.md for usage examples.")


if __name__ == "__main__":
    main()