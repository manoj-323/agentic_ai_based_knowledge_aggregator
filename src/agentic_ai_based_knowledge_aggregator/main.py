#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crew import AgenticAiBasedKnowledgeAggregator

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run(inp, _field):
    """
    Run the crew.
    """
    inputs = {
        '_input': inp,
        '_field': _field,
    }

    result = None
    try:
        result = AgenticAiBasedKnowledgeAggregator().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

    print(result)


inp = "help me pick my first bike"
_field = "motorcycles"
run(inp, _field)