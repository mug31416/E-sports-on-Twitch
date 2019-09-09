#!/usr/bin/env bash


for i in A B C D; do
    for j in 1 2 3 4 5 6 7 8 9 10; do
        wget "https://play.esea.net/index.php?s=stats&d=ranks&rank=${i}&page=${j}";
    done;
done