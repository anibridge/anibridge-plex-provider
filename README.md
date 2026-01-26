# anibridge-plex-provider

An [AniBridge](https://github.com/anibridge/anibridge) provider for [Plex](https://www.plex.tv/).

_This provider comes built-in with AniBridge, so you don't need to install it separately._

## Configuration

### `url` (`str`)

The base URL of the Plex server (e.g., http://localhost:32400).

### `token` (`str`)

The account API token of the Plex server admin. Get a token by following [these instructions](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/).

### `user` (`str`)

The Plex user to synchronize. This can be a username, email, or display name.

### `sections` (`list[str]`, optional)

A list of Plex library section names to constrain synchronization to. Leave empty/unset to include all sections.

### `genres` (`list[str]`, optional)

A list of genres to constrain synchronization to. Leave empty/unset to include all genres.

### `strict` (`bool`, optional, default: `true`)

Whether to enforce strict matching when resolving mappings. If `true`, only exact mapping matches of a show's episode ordering (TMDB or TVDB) will be considered. If `false`, falling back from TMDB to TVDB (or vice versa) is allowed.

You can configure episode ordering in the show's or section's 'Advanced' settings.

```yaml
library_provider_config:
  plex:
    url: ...
    token: ...
    user: ...
    # sections: []
    # genres: []
    # strict: true
```
