#!/bin/bash                                                                                                                                                                   

contentArray=(BeingRural5.0 agf5.0 Nancarrow5.0)
styleArray=(BeingRural5.0 agf5.0 Nancarrow5.0)

for content in ${contentArray[@]}
do
    for style in ${styleArray[@]}
    do
	if [ "$style" == "$content" ]
	then
	   continue
	fi
	echo $content $style
    done
done

