"""Fail-closed slot for the one permitted integrated Stage B candidate.

This file becomes runnable only after Route 2 and Route 3 produce isolated,
hash-bound full-validation parent artifacts. Until then it deliberately has no
model and is not present in the candidate manifest.
"""

candidate_id = 'integrated-stage-b'
optimization_route = 'accuracy-first'
auto_run = False
blocked_reason = 'isolated structural and numeric Stage B parents are required'
integration_spec = dict(features=(), parent_results=())
model = None

