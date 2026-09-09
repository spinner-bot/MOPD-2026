#!/usr/bin/env python3
"""
Simple test to verify probe logging code is syntactically correct and importable
"""

import sys
import os

# Add the code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("Testing probe logging code...")

try:
    # Test 1: Import the modified dp_actor module
    print("Test 1: Importing dp_actor module...")
    from verl.workers.actor import dp_actor
    print("✓ dp_actor imported successfully")

    # Test 2: Check if DataParallelPPOActor class exists
    print("\nTest 2: Checking DataParallelPPOActor class...")
    assert hasattr(dp_actor, 'DataParallelPPOActor'), "DataParallelPPOActor not found"
    print("✓ DataParallelPPOActor class exists")

    # Test 3: Check if _run_probe_logging method exists
    print("\nTest 3: Checking _run_probe_logging method...")
    assert hasattr(dp_actor.DataParallelPPOActor, '_run_probe_logging'), "_run_probe_logging method not found"
    print("✓ _run_probe_logging method exists")

    # Test 4: Check if update_policy method calls probe logging
    print("\nTest 4: Checking update_policy integration...")
    import inspect
    source = inspect.getsource(dp_actor.DataParallelPPOActor.update_policy)
    assert '_run_probe_logging' in source, "update_policy doesn't call _run_probe_logging"
    print("✓ update_policy calls _run_probe_logging")

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
    print("\nProbe logging code is correctly integrated.")

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
