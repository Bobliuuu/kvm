# Sourced by the scripts that run on a shared GPU box (not executable on its own). Needs HOST and SSH set by the caller.
#   kvm_scratch NAME   sets REMOTE: $KVM_SCRATCH if given (shared on purpose; the caller owns it), else a private
#                      ~/kvm-scratch-NAME-<user>-<pid> that nobody else's rsync can overwrite.
#   kvm_scratch_done   removes an auto-named REMOTE (not $KVM_SCRATCH; skipped with KVM_KEEP_SCRATCH=1). Call it on success only,
#                      so a failed run keeps its directory for debugging.
#   kvm_lock SECONDS   sets FLK to the GPU-lock prefix; empty when KVM_NO_LOCK is non-empty (an outer hold, e.g. run_all.sh, exists).
# rsync into REMOTE never uses --delete: a directory somebody else may share is only ever added to.
kvm_scratch() {
  if [ -n "${KVM_SCRATCH:-}" ]; then REMOTE="$KVM_SCRATCH"; KVM_SCRATCH_AUTO=0
  else REMOTE="~/kvm-scratch-$1-${USER:-$(id -un)}-$$"; KVM_SCRATCH_AUTO=1; fi
}
kvm_scratch_done() {
  if [ "${KVM_SCRATCH_AUTO:-0}" = 1 ] && [ "${KVM_KEEP_SCRATCH:-0}" != 1 ]; then
    case "$REMOTE" in "~/kvm-scratch-"*) $SSH "$HOST" "rm -rf $REMOTE" || true;; esac
  else echo "remote scratch kept: $REMOTE"; fi
}
kvm_lock() { FLK="flock -w $1 ~/.gpu.lock"; [ -z "${KVM_NO_LOCK:-}" ] || FLK=""; }
