"""Plex provider configuration."""

from typing import Annotated

import msgspec


class PlexProviderConfig(msgspec.Struct, kw_only=True):
    """Configuration for the Plex provider."""

    url: Annotated[
        str,
        msgspec.Meta(description="The base URL of the Plex server."),
    ]
    token: Annotated[
        str,
        msgspec.Meta(description="The Plex authentication token for the target user."),
    ]
    home_user: (
        Annotated[
            str,
            msgspec.Meta(
                description=(
                    "Optional Plex home user identifier. "
                    "Only used when the provided token belongs to a Plex Home admin."
                )
            ),
        ]
        | None
    ) = None
    sections: Annotated[
        list[str],
        msgspec.Meta(
            description=(
                "A list of Plex library section names to constrain synchronization to."
            )
        ),
    ] = msgspec.field(default_factory=list)
    genres: Annotated[
        list[str],
        msgspec.Meta(description="A list of genres to constrain synchronization to."),
    ] = msgspec.field(default_factory=list)
    strict: Annotated[
        bool,
        msgspec.Meta(
            description="Whether to enforce strict matching when resolving mappings."
        ),
    ] = True
