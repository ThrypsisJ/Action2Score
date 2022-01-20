#!/bin/bash

for server in kr na1 jp1 euw1 eun1
do
    gnome-terminal --title $server --tab -- python3 timeline_extractor.py $server 
done