sudo sysctl kernel.perf_event_paranoid=1

poetry shell
nsys profile --trace cuda,osrt,nvtx --gpu-metrics-devices all --cuda-memory-usage true --force-overwrite true --output profile_run_v1 python test.py