#!/usr/bin/env bash
#
# Downloads FLORES-200 and NTREX.

set -euo pipefail

flores_url=https://tinyurl.com/flores200dataset
ntrex_url=https://github.com/MicrosoftTranslator/NTREX.git

ntrex_folder=data2/ntrex
flores_folder=data2/flores
flores_gzip=flores.tar.gz

# make data/{flores,ntrex}
mkdir -vp $flores_folder $ntrex_folder

# download flores
#wget \
    #--trust-server-names \
    #--output-document "${flores_folder}/${flores_gzip}" \
    #"${flores_url}"

pushd "${flores_folder}" && tar xzvf "${flores_gzip}" && popd

# download ntrex and remove associated .git
git clone "${ntrex_url}" "${ntrex_folder}"
rm -vrf "${ntrex_folder}/.git"
