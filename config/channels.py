"""
channels.py — Channel configuration for MultiChannelOrchestrator.

Each ChannelConfig defines one YouTube channel with its credentials,
content category, daily upload quota, and output isolation settings.
"""

from dataclasses import dataclass


@dataclass
class ChannelConfig:
    """Configuration for a single YouTube channel."""

    channel_key: str        # Matches .credentials/{channel_key}_token.json
    name: str               # Human-readable channel name
    channel_id: str = ""            # YouTube channel ID (UC…)
    handle: str = ""                # YouTube handle (@…)
    category: str = "success"       # Default content category
    daily_quota: int = 3            # Max uploads per day
    timezone: str = "Africa/Lagos"
    output_dir: str = ""            # Defaults to data/output/{channel_key}/ if empty

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = f"data/output/{self.channel_key}"


# ---------------------------------------------------------------------------
# Channel definitions — edit to add or remove channels
# ---------------------------------------------------------------------------

CHANNELS: list[ChannelConfig] = [
    ChannelConfig(
        channel_key="default",
        name="Stoic Wisdom",
        category="success",
        daily_quota=3,
        timezone="Africa/Lagos",
    ),
    ChannelConfig(
        channel_key="career",
        name="Career Accelerator",
        category="career",
        daily_quota=2,
        timezone="Africa/Lagos",
    ),
    ChannelConfig(
        channel_key="money_debate",
        name="Money Heresy",
        channel_id="UC9nKSmjC4g9QEEbPHVQEh6g",
        handle="@moneyheresy",
        category="money",
        daily_quota=3,
        timezone="Africa/Lagos",
    ),
]
