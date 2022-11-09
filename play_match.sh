#!/usr/bin/env bash

THIS_DIR=$( cd "$( dirname "$0" )" && pwd )

match_name=$1

cd $THIS_DIR/third_party/project_acpc_server && ./play_match.pl \
$match_name $THIS_DIR/holdem.limit.2p.game $2 0 \
Alice $THIS_DIR/$3 Bob $THIS_DIR/$4

cd $THIS_DIR && python post_match.py -f $THIS_DIR/third_party/project_acpc_server/$match_name.log