#!/bin/bash
curl -c /tmp/wlt "202.38.64.59/cgi-bin/ip?cmd=login&name=zjxkz20170625&password=161734" > /dev/null
curl -b /tmp/wlt "202.38.64.59/cgi-bin/ip?cmd=set&type=0&exp=0" > /dev/null
rm /tmp/wlt