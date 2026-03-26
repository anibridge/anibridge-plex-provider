"""Plex provider configuration."""

from pydantic import BaseModel, ConfigDict, Field


class PlexProviderConfig(BaseModel):
    """Configuration for the Plex provider."""

    url: str = Field(default=..., description="The base URL of the Plex server.")
    token: str = Field(
        default=..., description="The Plex authentication token for the target user."
    )
    home_user: str | None = Field(
        default=None,
        description=(
            "Optional Plex home user identifier. "
            "Only used when the provided token belongs to a Plex Home admin."
        ),
        validation_alias="user",
    )
    sections: list[str] = Field(
        default_factory=list,
        description=(
            "A list of Plex library section names to constrain synchronization to."
        ),
    )
    genres: list[str] = Field(
        default_factory=list,
        description="A list of genres to constrain synchronization to.",
    )
    strict: bool = Field(
        default=True,
        description="Whether to enforce strict matching when resolving mappings.",
    )

    model_config = ConfigDict(populate_by_name=True)
