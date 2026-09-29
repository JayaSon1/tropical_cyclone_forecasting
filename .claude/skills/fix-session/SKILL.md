---
name: fix-session
description: Run one fix session from AUDIT.md section 8, test-first. Use when Jaya types /fix-session N.
disable-model-invocation: true
---
Run fix session $ARGUMENTS from AUDIT.md section 8.

## 1. Plan first (no edits yet)
- Read AUDIT.md (section 8 for this session, plus the related risks and
  decisions) and CLAUDE.md.
- Present a short plan: the goal, the tests to write (with what each
  asserts), the files to change, and whether this session changes
  data/model/metrics.
- Flag anything ambiguous as a question. WAIT for my approval.

## 2. Tests first
- Write the tests. Run them and SHOW me they fail on the current code
  (or explain why a test legitimately passes already).
- During steps 2-3, loop with the fast suite (skips bootstrap-heavy tests):
  PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe -m pytest tests -q -p no:cacheprovider -m "not slow"
  If a new test is itself marked slow, run it by name to show it fail and pass.

## 3. Smallest change
- Make the smallest change that makes the tests pass.
- Touch nothing outside this session's scope. If you find a problem
  outside scope, note it in your summary instead of fixing it.
- If a decision is needed that AUDIT.md doesn't cover, STOP and ask.

## 4. Verify
- Run the full suite, including slow tests:
  PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
  and show the output.
- The golden-metrics test must still pass (re-baselining needs my approval).
- Confirm nothing in artifacts/ or results/ was written (show modification
  dates) unless this session is the retrain.
- Group any NEW warnings by source; flag any from our own code.

## 5. Summarise, don't commit
- What changed (files + one line each), test counts before/after,
  anything you were unsure of, and any out-of-scope issues found.
- Do NOT commit. Wait for me to run /ship-session.

Never change or delete a test just to make it pass - stop and ask me.
