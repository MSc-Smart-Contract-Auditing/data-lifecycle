#!/bin/bash

if brew list -1 | grep -q "sshpass"; then
    brew upgrade sshpass
else
    echo "sshpass is not installed. Installing sshpass using Homebrew..."
    brew install sshpass
fi

clear

USERNAME="${ICL_USERNAME}"
PASSWORD="${ICL_PASSWORD}"

sshpass -p "$PASSWORD" ssh -t $USERNAME@shell2.doc.ic.ac.uk '\
    ssh -t gpucluster2 "salloc --gres=gpu:1; bash"\
'

echo "Keep this window OPEN to maintain the connection."