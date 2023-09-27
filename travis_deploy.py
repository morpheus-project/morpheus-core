# MIT License
# Copyright 2020 Ryan Hausen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ==============================================================================
"""Used in a Travis build to deploy pypi

BE CAREFUL WHEN ADDING PRINT STATEMENTS TO THIS SCRIPT!!!!
You can leak secrets to the travis log.

This performs the following steps:

1. Get the current version of the package from morpheus_core.__version__.py
2. Increment the microversion by 1 and save to morpheus_core.__version__.py
3. Run setup.py sdist bdist_wheel
4. Push to pypi with twine
5. Git tag build with version and push
"""

import os
from collections import namedtuple

GIT_TRAVIS_UNAME = "morpheus-travis"
PYPI_TRAVIS_UNAME = "morpheus-travis"
TRAVIS_PWD = os.environ["TRAVIS_PWD"]
TRAVIS_PYPI_PWD = os.environ["TRAVIS_PYPI_PWD"]

LOCAL = os.environ.get("TRAVIS_BUILD_DIR")

PACKAGE = "morpheus_core"
REPO = "morpheus-core"

Version = namedtuple("Version", ["major", "minor", "micro"])


def get_version():
    version_file = os.path.join(LOCAL, f"{PACKAGE}/__version__.py")
    with open(version_file, "r") as f:
        major, minor, micro = [
            int(v) for v in f.readlines()[0].strip().replace('"', "").split(".")
        ]

    micro += 1

    with open(version_file, "w") as f:
        f.write(f'"""{major}.{minor}.{micro}"""')

    ver = Version(major=major, minor=minor, micro=micro)

    print(f"Deploying version:{ver.major}.{ver.minor}.{ver.micro}")

    return ver


def deploy_pypi():
    dist_path = os.path.join(LOCAL, "dist")
    os.mkdir(dist_path)

    print("Running setup.py...")
    setup_loc = os.path.join(LOCAL, "setup.py")
    os.system(f"python {setup_loc} sdist bdist_wheel")

    print("Uploading via twine...")
    os.system(
        f"twine upload -u {PYPI_TRAVIS_UNAME} -p {TRAVIS_PYPI_PWD} "
        + os.path.join(dist_path, "*")
    )


# https://medium.com/even-financial-engineering/continuous-release-pipeline-with-travis-ci-78fc01febf38
def github_tag_and_push(ver: Version):
    print("Pushing version file to github")
    os.system('git config --global user.email "ryanhausen@gmail.com"')
    os.system('git config --global user.name "Morpheus TravisBot"')
    os.system(
        f'git commit -a -m "[skip travis] TRAVIS:Setting version to {ver.major}.{ver.minor}.{ver.micro}"'
    )
    os.system(
        f"git push 'https://{GIT_TRAVIS_UNAME}:{TRAVIS_PWD}@github.com/morpheus-project/{REPO}.git' HEAD:master"
    )

    print("Pushing tag to github")
    os.system(f"git tag v{ver.major}.{ver.minor}.{ver.micro}")
    os.system(
        f"git push --tags 'https://{GIT_TRAVIS_UNAME}:{TRAVIS_PWD}@github.com/morpheus-project/{REPO}.git'"
    )


def main():
    ver = get_version()

    github_tag_and_push(ver)
    deploy_pypi()

    print("done!")


if __name__ == "__main__":
    main()
