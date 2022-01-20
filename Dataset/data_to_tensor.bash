#!/bin/bash

for server in kr na1 jp1 euw1 eun1
do
    gnome-terminal --title $server --tab -- python3 data_to_tensor.py $server 
done