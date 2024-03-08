#!/bin/bash

git_status=$(git status --porcelain)
if [[ $git_status ]]; then
  echo "Checkout code is not clean"
  echo "${git_status}"
  exit 1
fi

find \( -name '*.cpp' -or -name '*.h' -or -name '*.cc' \) -not -path "./external/*" -not -path "./external_libs/*" | xargs clang-format-13 -i -style=file
git_status=$(git status --porcelain)
if [[ $git_status ]]; then
  echo "clang-format-13 is not happy, please run \"clang-format-13 -i -style=file /PATH/TO/foo.cpp\" to the following files"
  echo "${git_status}"
  exit 1
else
  echo "PASSED C++ format"
fi
