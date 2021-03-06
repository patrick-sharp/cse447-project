#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Zack Barnes, zbarnes\nJonah Kolar, jakolar\nPatrick Sharp, sharp77" > submit/team.txt

# submit writup
cp project_report.docx submit/project_report.docx

# make predictions on example data submit it in pred.txt
python3 src/main.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit requirements.txt
cp requirements.txt submit/requirements.txt

# submit source code
rm -rf src/__pycache__
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r submit.zip submit
