import sys
import os
import traceback

print(f"CWD: {os.getcwd()}")
sys.path.append(os.getcwd())

try:
    print("Attempting to import evaluation.metrics...")
    from evaluation.metrics import SummarizationMetrics
    print("Import successful.")
    
    print("Attempting to instantiate SummarizationMetrics...")
    m = SummarizationMetrics()
    print("Instantiation successful.")
    
except Exception:
    traceback.print_exc()
