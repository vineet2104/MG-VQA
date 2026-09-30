"""
mgvqa.orchestrator — Multi-worker parallel execution for MG-VQA-Bench.

Spawns N worker subprocesses, distributes episodes round-robin,
monitors for crashes/recycling, and merges results post-hoc.
"""
