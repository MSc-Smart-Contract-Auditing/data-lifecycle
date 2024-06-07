#!/bin/bash

USERNAME="${ICL_USERNAME}"
PASSWORD="${ICL_PASSWORD}"

# Read command-line options
while getopts ":p:e:" opt; do
  case ${opt} in
    p )
      port=$OPTARG
      ;;
    e )
      env=$OPTARG
      ;;
    \? ) echo "Usage: ./run-server [-p port] [-e env]"
      exit 1
      ;;
  esac
done

if [ -z "${port}" ] || [ -z "${env}" ]; then
    echo "Both port and environment must be specified."
    echo "Usage: cmd [-p port] [-e env]"
    exit 1
fi

echo "Starting ${env} server on port ${port}..."
sshpass -p "$PASSWORD" ssh -tt -L "${port}:localhost:${port}" $USERNAME@shell3.doc.ic.ac.uk "/vol/linux/bin/slurm_sshtojob.sh -g -w ~/ -p ${port} -e /vol/bitbucket/kza23/${env}"