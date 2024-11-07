import pooch


def download(url: str, path=None, known_hash=None, progressbar=True):
    return pooch.retrieve(
        url,
        known_hash=known_hash,
        path=path,
        progressbar=progressbar,
    )
