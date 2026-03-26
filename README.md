# anibridge-plex-provider

An [AniBridge](https://github.com/anibridge/anibridge) provider for [Plex](https://www.plex.tv/).

_This provider comes built-in with AniBridge, so you don't need to install it separately._

## Configuration

```yaml
library_provider_config:
  plex:
    url: ...
    token: ...
    # home_user: ...
    # sections: []
    # genres: []
    # strict: true
```

### `url`

`str` (required)

The base URL of the Plex server (e.g., http://localhost:32400).

### `token`

`str` (required)

The Plex authentication token for the target user being synchronized. Get a token by following [these instructions](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/).

### `home_user`

`str` (optional, default: `None`)

Optional Plex Home user to use. If the token provided belongs to the user you want to synchronize, you can leave this unset.

If set, the provider will attempt to switch to the specified Plex Home user using the Plex API. The account token provided must be the token of the Plex Home owner, otherwise the provider will fail to authenticate.

### `sections`

`list[str]` (optional, default: `[]`)

A list of Plex library section names to constrain synchronization to. Leave empty/unset to include all sections.

### `genres`

`list[str]` (optional, default: `[]`)

A list of genres to constrain synchronization to. Leave empty/unset to include all genres.

### `strict`

`bool` (optional, default: `True`)

Whether to enforce strict matching when resolving mappings. If `true`, only exact mapping matches of a show's episode ordering (TMDB or TVDB) will be considered. If `false`, falling back from TMDB to TVDB (or vice versa) is allowed.

You can configure episode ordering in the show's or section's 'Advanced' settings.
