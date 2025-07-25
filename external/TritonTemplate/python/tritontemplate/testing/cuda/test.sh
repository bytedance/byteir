#!/bin/bash

set -e
for file in *.py; do
    if [ -f "$file" ]; then
        echo "Running pytest on $file"
        GITHUB_CI_TEST=true pytest "$file"
    fi
done