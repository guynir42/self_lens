import os
import requests
import hashlib


def fetch_volumes(obj="WD", survey=None):
    """Make sure we have the effective volume data,
    if not, download it from Zenodo.

    Parameters
    ----------
    obj : str
        The type of object to fetch volumes for.
        Can be 'WD' or 'BH'. Default is 'WD'.
    survey: str or list of str
        The survey(s) to fetch volumes for.
        The default is to get all surveys.
    """

    if obj not in ["WD", "BH"]:
        raise ValueError('object must be "WD" or "BH"')

    if survey is not None:
        if isinstance(survey, str):
            surveys = [survey]
        elif isinstance(survey, list):
            surveys = survey
        else:
            raise ValueError("survey must be a string or list of strings")
    else:
        surveys = ["TESS", "ZTF", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY", "LAST"]

    # get the links from Zenodo API
    if obj == "WD":
        res = requests.get("https://zenodo.org/api/records/8340555")
    elif obj == "BH":
        res = requests.get("https://zenodo.org/api/records/8340930")

    j = res.json()

    for s in surveys:
        key = f"simulate_{s}_{obj}.nc"
        if os.path.isfile(key):
            print(f"File {key} already exists!")
            break
        for record in j["files"]:
            if record["key"] == key:
                url = record["links"]["self"]
                filename = os.path.basename(url)
                if not os.path.exists(filename):
                    print(f"Downloading {url} to {filename}")
                    res = requests.get(url)
                    with open(filename, "wb") as f:
                        f.write(res.content)
                    # check the MD5 hash and size of file:
                    if res.headers["Content-Length"] != record["size"]:
                        raise ValueError(f"File size mismatch for {key}")
                    md5 = hashlib.md5()
                    md5.update(res.content)

                    if record["checksum"] != md5.hexdigest():
                        raise ValueError(f"MD5 hash mismatch for {key}")
                break
        raise ValueError(f"Could not find {key} in Zenodo API response")


if __name__ == "__main__":
    fetch_volumes(obj="WD", survey="TESS")
