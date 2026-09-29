---
name: ship-session
description: Verify and commit a completed fix session on fix/pipeline. Use when Jaya types /ship-session N.
disable-model-invocation: true
---
Commit fix session $ARGUMENTS.

1. Confirm we are on branch fix/pipeline. If not, stop and tell me.
2. Re-run the full test suite and show the result. If anything fails, stop.
3. Show git status and git diff --stat. Check that no data files, .env,
   keys or files from artifacts/ or results/ are staged (unless this is
   the retrain session, and I've approved re-baselining).
4. Mark the session as done in AUDIT.md section 8 (with today's date).
5. Commit as one commit: "Fix session $ARGUMENTS: <what now works>",
   with a short body listing the AUDIT.md risks/decisions addressed.
6. Show git log --oneline -3. Don't push.
