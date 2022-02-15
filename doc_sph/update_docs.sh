#!/bin/bash

up=./
export PYTHONPATH=$up:$PYTHONPATH

source=doc_sph/source
target=doc_sph/_build/html

sphinx-versioning build $source $target 2>&1 | tee log.doc # && sshpass -p springpypi scp -P 10000 -r $target spring@10.10.40.93:~/doc_sph/up && echo "update up docs done"
