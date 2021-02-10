#!/bin/bash

python3 load_wiki.py
[ -d data ] || mkdir data
mv *.txt data
totalFiles=$(ls data | wc -l)
echo "Number of articles downloaded was $totalFiles"
