#!/usr/bin/env bash

THIS_DIR=$( cd "$( dirname "$0" )" && pwd )

cd $THIS_DIR/third_party/project_acpc_server && ./play_match.pl \
match_1 $THIS_DIR/leduc.limit.2p.game 10 0 \
Alice ./example_player.leduc.limit.2p.sh \
Bob $THIS_DIR/run_play_leduc.sh