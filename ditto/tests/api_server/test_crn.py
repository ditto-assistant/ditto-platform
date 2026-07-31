from uuid import uuid4

from ditto.api_server.crn import continual_anchor_horizon


def test_continual_anchor_horizon_extends_beyond_legacy_cap() -> None:
    first = uuid4()
    second = uuid4()

    assert continual_anchor_horizon({}) == 16
    assert (
        continual_anchor_horizon(
            {
                first: range(16),
                second: (*range(16), 99),
            }
        )
        == 18
    )
