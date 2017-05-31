#!/bin/bash                                                                                                                                                                   

orientationArray=(height)

for orientation in ${orientationArray[@]}
do
    if [ "$orientation" == "channels" ]
    then
	l1channels=2048
    else
	l1channels=32
    fi
    echo "l1 channels is  $l1channels"

done
