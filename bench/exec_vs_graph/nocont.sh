#!/usr/bin/env bash
exec "$(dirname "$0")/../../kvm_exec" --no-cont "$@"
