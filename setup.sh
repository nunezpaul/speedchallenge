#!/usr/bin/env bash

mkdir data
wget sunlight.caltech.edu/pnunez/speedchallenge/test.mp4 && mv test.mp4 data
wget sunlight.caltech.edu/pnunez/speedchallenge/train.mp4 && mv train.mp4 data
wget sunlight.caltech.edu/pnunez/speedchallenge/train.txt && mv train.txt data
mkdir data/images