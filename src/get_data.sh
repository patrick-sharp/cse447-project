#!/bin/bash

python3 load_wiki.py
[ -d data ] || mkdir data
mv *.txt data
totalFiles=$(ls data | wc -l)
numMandarin=$(ls data/*zh.txt | wc -l)
numCantonese=$(ls data/*zh-yue.txt | wc -l)
numEnglish=$(ls data/*en.txt | wc -l)
numItalian=$(ls data/*it.txt | wc -l)
numGerman=$(ls data/*de.txt | wc -l)
numJaponese=$(ls data/*ja.txt | wc -l)
numRussian=$(ls data/*ru.txt | wc -l)
numSpanish=$(ls data/*es.txt | wc -l)
numFrench=$(ls data/*fr.txt | wc -l)
numNorwegian=$(ls data/*no.txt | wc -l)
numDutch=$(ls data/*nl.txt | wc -l)
numDanish=$(ls data/*da.txt | wc -l)
numSwedish=$(ls data/*sv.txt | wc -l)
echo "-------------- Finished Downloading --------------"
echo "Total number of articles downloaded was $totalFiles"
echo "Number of articles downloaded in Mandarin: $numMandarin"
echo "Number of articles downloaded in Cantonese: $numCantonese"
echo "Number of articles downloaded in English: $numEnglish"
echo "Number of articles downloaded in Italian: $numItalian"
echo "Number of articles downloaded in German: $numGerman"
echo "Number of articles downloaded in Japonese: $numJaponese"
echo "Number of articles downloaded in Russian: $numRussian"
echo "Number of articles downloaded in Spanish: $numSpanish"
echo "Number of articles downloaded in French: $numFrench"
echo "Number of articles downloaded in Norwegian: $numNorwegian"
echo "Number of articles downloaded in Dutch: $numDutch"
echo "Number of articles downloaded in Danish: $numDanish"
echo "Number of articles downloaded in Swedish: $numSwedish"
