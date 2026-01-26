"""Plex provider for AniBridge."""

import importlib.metadata
import os
import uuid

from anibridge_plex_provider.library import PlexLibraryProvider

__all__ = ["PlexLibraryProvider"]

# The below environment variables are consumed by the python-plexapi library
# and are used to identify the client making the requests to the Plex server.
# Having a consistent identifier is important so that the server doesn't think
# the client is a new one every time it starts (which causes "New Device"
# notifications)
os.environ["PLEXAPI_HEADER_IDENTIFIER"] = uuid.uuid3(
    uuid.NAMESPACE_DNS, "AniBridge"
).hex
os.environ["PLEXAPI_HEADER_DEVICE_NAME"] = "AniBridge"
os.environ["PLEXAPI_HEADER_VERSION"] = importlib.metadata.version(
    "anibridge-plex-provider"
)
os.environ["PLEXAPI_HEADER_PROVIDES"] = ""
os.environ["PLEXAPI_PLEXAPI_AUTORELOAD"] = "0"
