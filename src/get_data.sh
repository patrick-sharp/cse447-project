#!/bin/bash

DATA_DIR='../data'
python3 load_wiki.py --data_dir ${DATA_DIR}
# [ -d data ] || mkdir data
# mv *.txt data
totalFiles=$(ls ${DATA_DIR} | wc -l)
numMandarin=$(ls ${DATA_DIR}/*zh.txt | wc -l)
numCantonese=$(ls ${DATA_DIR}/*zh-yue.txt | wc -l)
numEnglish=$(ls ${DATA_DIR}/*en.txt | wc -l)
numItalian=$(ls ${DATA_DIR}/*it.txt | wc -l)
numGerman=$(ls ${DATA_DIR}/*de.txt | wc -l)
numJapanese=$(ls ${DATA_DIR}/*ja.txt | wc -l)
numRussian=$(ls ${DATA_DIR}/*ru.txt | wc -l)
numSpanish=$(ls ${DATA_DIR}/*es.txt | wc -l)
numFrench=$(ls ${DATA_DIR}/*fr.txt | wc -l)
numNorwegian=$(ls ${DATA_DIR}/*no.txt | wc -l)
numDutch=$(ls ${DATA_DIR}/*nl.txt | wc -l)
numDanish=$(ls ${DATA_DIR}/*da.txt | wc -l)
numSwedish=$(ls ${DATA_DIR}/*sv.txt | wc -l)
echo "-------------- Finished Downloading --------------"
echo "Total number of articles downloaded was     $totalFiles"
echo "Number of articles downloaded in Mandarin:  $numMandarin"
echo "Number of articles downloaded in Cantonese: $numCantonese"
echo "Number of articles downloaded in English:   $numEnglish"
echo "Number of articles downloaded in Italian:   $numItalian"
echo "Number of articles downloaded in German:    $numGerman"
echo "Number of articles downloaded in Japanese:  $numJapanese"
echo "Number of articles downloaded in Russian:   $numRussian"
echo "Number of articles downloaded in Spanish:   $numSpanish"
echo "Number of articles downloaded in French:    $numFrench"
echo "Number of articles downloaded in Norwegian: $numNorwegian"
echo "Number of articles downloaded in Dutch:     $numDutch"
echo "Number of articles downloaded in Danish:    $numDanish"
echo "Number of articles downloaded in Swedish:   $numSwedish"
