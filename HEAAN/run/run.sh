#!/bin/sh

# Parse options
while getopts "rcd" opt; do
    case $opt in
        r)
            echo "Rebuilding libHEAAN... (cd to ../lib)"
            cd ../lib
            make clean
            make
            cd ../run
            echo "Rebuilding done (cd to ../run)"
            ;;
        k)
            echo "Cleaning Public Key ..."
            rm ./serkey/*
            ;;
        c)
            echo "Cleaning Ciphers Saved..."
            rm ./cipher/*
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
    esac
done

# compile and run the new program
make clean
make new
./new