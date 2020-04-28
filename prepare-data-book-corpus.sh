# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Usage: ./get-data-xnli.sh
#

set -e

# data paths
MAIN_PATH=$PWD
OUTPATH=$PWD/data/processed
DATA_PATH=$PWD/data

# tools paths
TOOLS_PATH=$PWD/tools
#TOKENIZER=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py

PROCESSED_PATH=$PWD/data/processed
#CODES_PATH=$MAIN_PATH/codes/codes_xnli_100
#VOCAB_PATH=$MAIN_PATH/codes/codes_xnli_100
#FASTBPE=$TOOLS_PATH/fastBPE/fast


# install tools: done independently this time
#sh ./install-tools.sh

# create directories
mkdir -p $OUTPATH
mkdir -p $PROCESSED_PATH

# download data
#sh ./wmt.sh

# Lower casing each line
echo "*** Preparing Corpus data ***"
for lg in en ; do
    awk -F '\n' '{ print $1}' $DATA_PATH/goblet_book.txt \
    | awk '{gsub(/\"/,"")};1' \
    | python $LOWER_REMOVE_ACCENT \
    > $PROCESSED_PATH/harry_potter.tok.$lg
done

for lg in en ; do
    awk -F '\n' '{ print $1}' $DATA_PATH/the_republic.txt \
    | awk '{gsub(/\"/,"")};1' \
    | python $LOWER_REMOVE_ACCENT \
    > $PROCESSED_PATH/the_republic.tok.$lg
done

for lg in ru ; do
    awk -F '\n' '{ print $1}' $DATA_PATH/multilingual/harry_potter_ru.txt \
    | awk '{gsub(/\"/,"")};1' \
    | python $LOWER_REMOVE_ACCENT \
    > $PROCESSED_PATH/harry_potter.tok.$lg
done

echo 'Finished preparing data.'