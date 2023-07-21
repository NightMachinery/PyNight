import os
import json
import glob
import subprocess
from typing import List, Dict
from importlib.metadata import distribution
from pynight.common_dict import simple_obj
from icecream import ic

##
def package_commit_get(
    package_name: str,
    import_p: bool = True,
    drop_nones=True,
) -> Dict[str, List[str]]:
    package_version = None
    package_dir = None
    if import_p:
        # Import the package dynamically
        package = __import__(package_name)

        # Get the directory of the package
        package_dir = os.path.dirname(package.__file__)

        # Try to get package version
        try:
            package_version = package.__version__
        except AttributeError:
            pass
    else:
        # Get site-packages path
        site_packages_path = distribution(package_name).locate_file("")

        # Construct path to .pth file
        pth_file_path = os.path.join(site_packages_path, f"{package_name}.pth")

        # Check if .pth file exists
        if os.path.isfile(pth_file_path):
            # Read .pth file to get actual package directory
            with open(pth_file_path, "r") as f:
                package_dir = f.read().strip()
        else:
            # .pth file not found, try direct_url.json
            dist_info_paths = glob.glob(
                os.path.join(
                    site_packages_path, f"{package_name}*.dist-info", "direct_url.json"
                )
            )
            if not dist_info_paths:
                package_dir = os.path.join(site_packages_path, package_name,)
                if not os.path.exists(os.path.join(package_dir, "__init__.py")):
                    raise Exception(
                        f"Neither .pth file nor direct_url.json file nor __init__.py file found for the package: {package_name}"
                    )
            else:
                # Read the first direct_url.json file to get actual package directory
                with open(dist_info_paths[0], "r") as f:
                    dir_info = json.load(f)
                    package_dir = dir_info.get("url").replace("file://", "")

    # Try to get the git commit hash
    try:
        commit = subprocess.check_output(
            ["git", "-C", package_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        commit = commit.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        commit = None

    # Try to get the list of dirty files
    try:
        dirty_files = subprocess.check_output(
            ["git", "-C", package_dir, "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        dirty_files = dirty_files.decode("utf-8").splitlines()
    except subprocess.CalledProcessError:
        dirty_files = None

    return simple_obj(
        _drop_nones=drop_nones,
        package_name=package_name,
        package_dir=package_dir,
        commit=commit,
        dirty_files=dirty_files,
        version=package_version,
    )


def packages_commit_get(packages, **kwargs):
    output = dict()
    for pkg in packages:
        try:
            res = vars(package_commit_get(pkg, **kwargs))
            output[pkg] = res
        except Exception as e:
            print(e)
            pass

    return output
##
