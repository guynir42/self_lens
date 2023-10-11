import os
import requests
import hashlib

SAVED_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved")
MATRICES_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "matrices")

# note: this is not actually needed anymore since we have added the matrices to the repo itself
DROPBOX_LINKS_SOURCE_MATRICES = [
    "https://www.dropbox.com/scl/fi/51tbcrmvqltq06rr2zayq/matrix_SR0.010-0.100_D0.000-3.000.npz?rlkey=iprljqthmhsffmht6nd0i98ht&dl=1",
    "https://www.dropbox.com/scl/fi/iqyjimtgfrw35c1j34o39/matrix_SR0.010-0.100_D0.000-3.000.txt?rlkey=uvrsyyaohg0wxe6yv10own1qo&dl=1",
    "https://www.dropbox.com/scl/fi/92gxty9p2qq9q0bn3bflj/matrix_SR0.100-1.000_D0.000-10.000.npz?rlkey=2lbd4plv0i0ry1teufmmbpdv2&dl=1",
    "https://www.dropbox.com/scl/fi/rzy098bq4cpafow1jb7b9/matrix_SR0.100-1.000_D0.000-10.000.txt?rlkey=3733ln9azzhn2ffzghcnxn696&dl=1",
    "https://www.dropbox.com/scl/fi/r0ge4t3vjvkjlbi5kh9ir/matrix_SR1.000-5.000_D0.000-20.000.npz?rlkey=zzabfyc7e0ccbc80wdcygshwb&dl=1",
    "https://www.dropbox.com/scl/fi/rfbv0aiob92qcvyuvry9t/matrix_SR1.000-5.000_D0.000-20.000.txt?rlkey=9iakwmzk6tansfqhflhvexg66&dl=1",
]

DROPBOX_LINKS_WD = {
    "ZTF": "https://www.dropbox.com/scl/fi/j7bhlkjqvwukloo6scxpw/simulate_ZTF_WD.nc?rlkey=kouihvedug9ijmoh1w428f5ug&dl=1",
    "TESS": "https://www.dropbox.com/scl/fi/y8clk772wtuc29nnjm16t/simulate_TESS_WD.nc?rlkey=h9e9jofr8sq6xu5qmybqodw6a&dl=1",
    "LSST": "https://www.dropbox.com/scl/fi/w42so01st79lfq4jzeegq/simulate_LSST_WD.nc?rlkey=fx733armzk8tmvc7crfucsf7e&dl=1",
    "DECAM": "https://www.dropbox.com/scl/fi/w29z0jobqw49pszfjvu2o/simulate_DECAM_WD.nc?rlkey=ths3th4l2qbggumu0fv9othip&dl=1",
    "CURIOS": "https://www.dropbox.com/scl/fi/4vcz7e24r81hqgwkjsrh5/simulate_CURIOS_WD.nc?rlkey=cssa5ygjqfmwv2tjblq5itoaz&dl=1",
    "CURIOS_ARRAY": "https://www.dropbox.com/scl/fi/kyfowkp8w01r7jgdcec5e/simulate_CURIOS_ARRAY_WD.nc?rlkey=np05nm18gc3n5us63gqf23j43&dl=1",
    "LAST": "https://www.dropbox.com/scl/fi/q94218hrkreahhk7gnh6z/simulate_LAST_WD.nc?rlkey=jm0n3y0fsyqbjpqa4l1rylp5k&dl=1",
}

DROPBOX_LINKS_BH = {
    "ZTF": "https://www.dropbox.com/scl/fi/jzzwuub7925veavn777q3/simulate_ZTF_BH.nc?rlkey=ey69ydhazy6pjuexmlh11w39d&dl=1",
    "TESS": "https://www.dropbox.com/scl/fi/91wbqzlcq498486l8dcy5/simulate_TESS_BH.nc?rlkey=utvifirx3kytz8ipvi8kz4z00&dl=1",
    "LSST": "https://www.dropbox.com/scl/fi/50g806rerxrawqgfv5ac6/simulate_LSST_BH.nc?rlkey=gmlbx4m4ht2khu2wwhb6ldgt4&dl=1",
    "DECAM": "https://www.dropbox.com/scl/fi/mvbcb7i361o2gqkbh5z44/simulate_DECAM_BH.nc?rlkey=lckhlzljoss2j2e0nxs5msq9a&dl=1",
    "CURIOS": "https://www.dropbox.com/scl/fi/f97fgm6b20wajnlk8znsi/simulate_CURIOS_BH.nc?rlkey=ks3y0ocho97sv4q9k5bqbbreh&dl=1",
    "CURIOS_ARRAY": "https://www.dropbox.com/scl/fi/jxzvvha1t0tezvoyy0ri1/simulate_CURIOS_ARRAY_BH.nc?rlkey=d42we7ychh623hpo7wolj5sol&dl=1",
    "LAST": "https://www.dropbox.com/scl/fi/2idhlgxcv1o47twryme95/simulate_LAST_BH.nc?rlkey=jl0jj0k9flkid0vfmhudx982g&dl=1",
}

DROPBOX_LINKS = dict(WD=DROPBOX_LINKS_WD, BH=DROPBOX_LINKS_BH, matrices=DROPBOX_LINKS_SOURCE_MATRICES)


def fetch_volumes(obj="WD", survey=None, verbose=False):
    """Make sure we have the effective volume data,
    if not, download it from dropbox or Zenodo.

    Parameters
    ----------
    obj : str
        The type of object to fetch volumes for.
        Can be 'WD' or 'BH'. Default is 'WD'.
    survey: str or list of str
        The survey(s) to fetch volumes for.
        The default is to get all surveys.
    verbose : bool
        Whether to print out the progress of the download.
    """
    obj = obj.upper()
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

    surveys = [s.upper() for s in surveys]
    success = {s: False for s in surveys}

    if not os.path.isdir(SAVED_FOLDER):
        os.mkdir(SAVED_FOLDER)

    # try to get this from dropbox first:
    for s in surveys:
        key = f"simulate_{s}_{obj}.nc"
        out_filename = os.path.join(SAVED_FOLDER, key)
        if os.path.isfile(out_filename):
            if verbose:
                print(f"File {out_filename} already exists!")
            success[s] = True
            continue

        try:  # try to get the dropbox link (this is much faster but requires me to keep those links alive forever)
            url = DROPBOX_LINKS[obj][s]
            if verbose:
                print(f'Using link "{url}" to download {key} to {out_filename}"')
            res = requests.get(url, allow_redirects=True)
            with open(out_filename, "wb") as f:
                f.write(res.content)
            success[s] = True

        except:
            pass  # let's just skip the dropbox link if there's a failure

    if all(success.values()):
        return  # skip this next part if we got all surveys already

    # get the links from Zenodo API
    if obj == "WD":
        zenodo_response = requests.get("https://zenodo.org/api/records/8340555")
    elif obj == "BH":
        zenodo_response = requests.get("https://zenodo.org/api/records/8340930")

    zenodo_json = zenodo_response.json()

    for s in surveys:
        key = f"simulate_{s}_{obj}.nc"
        out_filename = os.path.join(SAVED_FOLDER, key)
        if os.path.isfile(out_filename):
            if verbose:
                print(f"File {out_filename} already exists!")
            success[s] = True
            continue

        for record in zenodo_json["files"]:
            if record["key"] == key:
                url = record["links"]["self"]
                if verbose:
                    print(f"Downloading {url} to {out_filename}")
                res = requests.get(url)

                # check the MD5 hash and size of file:
                if int(res.headers["Content-Length"]) != record["size"]:
                    raise ValueError(
                        f"File size mismatch for {key}: {res.headers['Content-Length']} vs {record['size']}"
                    )
                md5 = hashlib.md5()
                md5.update(res.content)

                if record["checksum"] != "md5:" + md5.hexdigest():
                    raise ValueError(f"MD5 hash mismatch for {key}")

                with open(out_filename, "wb") as f:
                    f.write(res.content)
                success[s] = True

                break  # break out of the loop over records
        else:
            raise ValueError(f"Could not find {key} in Zenodo API response")

    if not all(success.values()):
        raise ValueError(f"Could not download all volumes! Missing: {[s for s in surveys if not success[s]]}")


if __name__ == "__main__":
    fetch_volumes(obj="WD", survey="TESS", verbose=True)
