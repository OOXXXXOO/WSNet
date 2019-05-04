#!/usr/bin/env bash
echo "***************************************"
echo "**************Video transfrom to Images************"
echo "Video file Name is $1"
source activate MLenv
ffmpeg -i $1 -r 0.01 -ss 00:00:00 -t 00:19:00 %03d.png
