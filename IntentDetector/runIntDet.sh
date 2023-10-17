#!/bin/bash

if [[ "${ACT}" == "train" ]]
then
  if [[ "${DO_XVAL}" == "1" ]]
  then
    python src/IntDetServiceRunner.py train "${DATAFILE}" "${MODEL_PREFIX}" "${VECTORIZER_ADDRESS}" "${VECTORIZER_PORT}" --xval
  else
    python src/IntDetServiceRunner.py train "${DATAFILE}" "${MODEL_PREFIX}" "${VECTORIZER_ADDRESS}" "${VECTORIZER_PORT}"
  fi
elif [[ "${ACT}" == "serve" ]]
then
  python src/IntDetServiceRunner.py serve "${MODEL_PREFIX}" "${SERVING_PORT}" "${VECTORIZER_ADDRESS}" "${VECTORIZER_PORT}"
else
  echo FAIL: ACT environment variable should be set to \"train\" or \"serve\"
fi
 
 