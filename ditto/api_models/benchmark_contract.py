"""Versioned benchmark prerequisites shared by ticket and artifact APIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkContract:
    version: int
    minimum_screening_policy_version: int
    requires_screened_image: bool


_CONTRACTS = {
    # v2 predates screened images and must remain source-build compatible for
    # validators which do not update during the rolling v3 activation.
    2: BenchmarkContract(2, 1, False),
    # A v3 dataset is only released after a policy-9 screener has produced an
    # archive whose complete bytes were verified by the platform.
    3: BenchmarkContract(3, 9, True),
    # v4 supersedes v3 without relaxing any prerequisite: same policy-9 screener
    # floor and the same verified-archive requirement.
    4: BenchmarkContract(4, 9, True),
}


def benchmark_contract(version: int) -> BenchmarkContract:
    """Return the immutable contract for ``version``; unknown versions fail closed."""
    try:
        return _CONTRACTS[version]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark version: {version}") from exc


def benchmark_contracts() -> tuple[BenchmarkContract, ...]:
    """Return every shipped contract in stable version order.

    Shipping code makes a contract *available*; it does not activate or open a
    rollout. The authenticated operator control uses this registry for target
    discovery so future benchmark bumps do not require another API hardcode.
    """
    return tuple(_CONTRACTS[version] for version in sorted(_CONTRACTS))


def latest_benchmark_contract() -> BenchmarkContract:
    """Return the newest contract shipped by this platform release."""
    return _CONTRACTS[max(_CONTRACTS)]
