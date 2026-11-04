"""kvmfe: Python frontend of the KVM megakernel framework.

model/program -> logical ops -> autodiff -> tile tasks -> dataflow edges -> workload JSON (see FRONTEND.md).
The scheduler below this layer knows nothing about models, training, serving or games.
"""
import logging

log = logging.getLogger('kvmfe')
