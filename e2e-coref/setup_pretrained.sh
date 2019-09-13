#!/bin/bash

mkdir -p ontonotes

function download_from_gcs() {
   curl -o $1 https://www.googleapis.com/storage/v1/b/e2e-coref/o/$1?alt=media
}
download_from_gcs ontonotes/char_vocab.english.txt

ckpt_file=c2f_final.tgz
download_from_gcs $ckpt_file
mkdir -p logs
tar -xzvf $ckpt_file -C logs
rm $ckpt_file
