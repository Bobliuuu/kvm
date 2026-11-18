# Sourced by the scripts that run on a shared GPU box (not executable on its own). Needs HOST and SSH set by the caller.
#   mvm_scratch NAME   sets REMOTE: $MVM_SCRATCH if given (shared on purpose; the caller owns it), else a private
#                      ~/mvm-scratch-NAME-<user>-<pid> that nobody else's rsync can overwrite.
#   mvm_scratch_done   removes an auto-named REMOTE (not $MVM_SCRATCH; skipped with MVM_KEEP_SCRATCH=1). Call it on success only,
#                      so a failed run keeps its directory for debugging.
#   mvm_lock SECONDS   sets FLK to the GPU-lock prefix; empty when MVM_NO_LOCK is non-empty (an outer hold, e.g. run_all.sh, exists).
# rsync into REMOTE never uses --delete: a directory somebody else may share is only ever added to.
mvm_scratch() {
  if [ -n "${MVM_SCRATCH:-}" ]; then REMOTE="$MVM_SCRATCH"; MVM_SCRATCH_AUTO=0
  else REMOTE="~/mvm-scratch-$1-${USER:-$(id -un)}-$$"; MVM_SCRATCH_AUTO=1; fi
}
mvm_scratch_done() {
  if [ "${MVM_SCRATCH_AUTO:-0}" = 1 ] && [ "${MVM_KEEP_SCRATCH:-0}" != 1 ]; then
    case "$REMOTE" in "~/mvm-scratch-"*) $SSH "$HOST" "rm -rf $REMOTE" || true;; esac
  else echo "remote scratch kept: $REMOTE"; fi
}
mvm_lock() { FLK="flock -w $1 ~/.gpu.lock"; [ -z "${MVM_NO_LOCK:-}" ] || FLK=""; }
