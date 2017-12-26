#!/bin/bash

function test {
    #time sleep 2s && echo "Done"
    python2 runner.py --alpha "$alpha" --log data/ydata_logs.txt data/webscope-articles.txt policy_linucb_concat.py
    #python2 runner.py --alpha "$alpha" --log data/webscope-logs.txt data/webscope-articles.txt policy_linucb_vec.py
}

IFS="
"; for i in `cat alphas.txt`; do
    for alpha in `echo $i | grep -Eo '[^ ]+'`; do
        test &
        export u_pid="`echo $!`"
    done
    echo $u_pid
    wait $u_pid
    echo "--- `date`"
done
