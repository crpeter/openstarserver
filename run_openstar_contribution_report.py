#!/usr/bin/env python3
"""Print live and historical contribution totals from a coordinator."""

import argparse

from openstar_coordinator_client import OpenStarCoordinatorClient


def _integer(value):
    return f"{int(value or 0):,}"


def _seconds(value):
    return f"{float(value or 0):,.3f} s"


def _rate(value):
    return "unavailable" if value is None else f"{float(value):,.3f}"


def print_scope(title, scope):
    print(f"\n{title}\n\nNetwork")
    print(f"  accepted work units: {_integer(scope['totalAcceptedWorkUnits'])}")
    print(f"  worker compute time: {_seconds(scope['totalWorkerComputeSeconds'])}")
    print(f"  Metal compute time: {_seconds(scope['totalMetalSeconds'])}")
    print(
        "  sample-frequency evaluations: "
        + _integer(scope["totalSampleFrequencyEvaluations"])
    )
    wall_rate = scope.get("sampleFrequencyEvaluationsPerWallSecond")
    if wall_rate is not None:
        print(
            "  network wall throughput (sample-frequency evaluations/sec): "
            + _rate(wall_rate)
        )
    print(
        "  aggregate Metal compute efficiency (evaluations/device-second): "
        + _rate(scope.get("aggregateSampleFrequencyEvaluationsPerMetalSecond"))
    )
    print("\nDevices")
    total = int(scope["totalSampleFrequencyEvaluations"] or 0)
    for device in scope["nodes"]:
        label = device.get("hardwareIdentifier") or device["nodeID"]
        print(f"\n{label}")
        print(f"  node: {device['nodeID']}")
        print(f"  platform: {device.get('platform') or 'unknown'}")
        print(f"  gpu: {device.get('gpuName') or 'unknown'}")
        print(f"  work units: {_integer(device['acceptedWorkUnits'])}")
        print(f"  Metal compute time: {_seconds(device['metalSeconds'])}")
        print(
            "  sample-frequency evaluations: "
            + _integer(device["sampleFrequencyEvaluations"])
        )
        print(
            "  sample-frequency evaluations/sec: "
            + _rate(device.get("sampleFrequencyEvaluationsPerMetalSecond"))
        )
        share = (100 * device["sampleFrequencyEvaluations"] / total) if total else 0
        print(f"  contribution share: {share:.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    summary = OpenStarCoordinatorClient(args.coordinator).contribution_summary()
    print("=== OpenStar contribution report ===")
    print(f"Coordinator session: {summary['coordinatorSessionID']}")
    print_scope("Current session", summary["currentSession"])
    print_scope("All time", summary["allTime"])


if __name__ == "__main__":
    main()
