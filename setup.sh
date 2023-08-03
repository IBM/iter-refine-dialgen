#!/bin/bash

#---------------------------------------------------------------------
#                                                                     
#  Licensed Materials -- Property of IBM
#
#  Restricted Materials of IBM
#                                                                     
#  (C) Copyright IBM Corporation 2017.  All Rights Reserved.
#
#  U.S. Government Users Restricted Rights:  Use, duplication or 
#  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.                                 
#                                                                     
#---------------------------------------------------------------------

function checkerror () {
    EXITCODE=$?
    MSG=$1
    if [ $EXITCODE -ne 0 ]; then
	echo $MSG
	exit $EXITCODE
    fi
}


# Make sure user is using bash
if [ -z $BASH ]; then
    echo "You must use bash"
    exit 1
fi


# Make sure use isn't just running this in a subshell - variables won't get
# exported properly.

prog=`echo $0 | sed -e 's/^-//'`
prog=`basename $prog`

#echo $prog

# Check that this is being sourced and not run
if [ $prog = "setup.sh" ]; then
    echo "source setup.sh, don't run it"
    exit 1
fi



# Determine NMTORCH directory
DIR=`dirname ${BASH_SOURCE}`
DIR=`realpath $DIR`

export PATH=$DIR:$PATH
export PYTHONPATH=$DIR:$PYTHONPATH
export NMT_HOME=$DIR

sep=`printf '=%.0s' {1..100}`
echo "$sep"
echo "setup script: ${BASH_SOURCE}"
DATE=`date`
echo "DATE: $DATE"
echo PYTHONPATH is $PYTHONPATH 1>&2
echo NMT_HOME is    $DIR 1>&2
echo WORK_DIR is    $PWD 1>&2
echo "$sep"

