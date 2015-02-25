#!/bin/sh

valgrind \
   --tool=memcheck \
   --leak-check=yes \
   --dsymutil=yes \
   --track-origins=yes \
   --error-limit=no \
   --suppressions=valgrind-python.supp \
   --num-callers=20 \
   -v \
   python $1

# This doesn't work
#valgrind \
#   --tool=drd \
#   --check-stack-var=yes \
#   --read-var-info=yes \
#   --suppressions=valgrind-python.supp \
#   --num-callers=20 \
#   -v \
#   python $1
