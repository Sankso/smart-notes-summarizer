
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from agent.executor import Executor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_summarization():
    executor = Executor()
    
    print("\n=== TEST 1: Short Text (Few-Shot) ===")
    short_text = """
    The Apollo program was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which succeeded in preparing and landing the first humans on the Moon from 1968 to 1972. It was first conceived in 1960 during the Eisenhower administration as a three-man spacecraft to follow the one-man Mercury project which would put the first Americans in space. Apollo was later dedicated to President John F. Kennedy's national goal of "landing a man on the Moon and returning him safely to the Earth" by the end of the 1960s, which he proposed in a May 25, 1961, address to Congress.
    """
    summary_short = executor.generate_summary(short_text, length="short")
    print(f"\nOriginal Length: {len(short_text)}")
    print(f"Summary: {summary_short}")
    
    print("\n=== TEST 2: Long Text (Refine Strategy) ===")
    # Generate long text > 2500 chars to trigger refine strategy
    long_text = short_text * 10 
    print(f"Long text length: {len(long_text)}")
    
    summary_long = executor.generate_summary(long_text, length="normal")
    print(f"\nRefined Summary: {summary_long}")

if __name__ == "__main__":
    test_summarization()
