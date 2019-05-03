tartupjupyter version 1.1

########### Licensing ####################
#
# Copyright (c) 2017, Pittsburgh Supercomputing Center
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of startupjupyter nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Pittsburgh Supercomputing Center    phone: 412.268.4960  fax: 412.268.5832
# 300 S. Craig Street            e-mail: remarks@psc.edu
# Pittsburgh, PA 15213            web: http://www.psc.edu/
#
######End of Licensing###################

# Software Purpose

# This is a bash script for launching a jupyter notebook from an interactive
# node. It assumes that the user has local ssh terminals on their desktop. 
# This works for GNU/Linux and MacOS users, but Windows does not have terminal
# support by default outside of Cygwin and Ubuntu Windows 10 Feature users.

#store hostname in a string
HOSTSTRPVT=`hostname`
HOSTSTR=$(echo $HOSTSTRPVT | sed 's/pvt/opa/g')
# if the user passes any command args at all, interpret it as a request for help

if [ -z $1 ]; then
# Only do this stuff on an interactive node. 
    if [[ "$HOSTSTR" = br00?.opa.bridges.psc.edu1 ]]; then
        echo "You are on" $HOSTSTR "which is a login node"
        echo "You must be on an interactive node to run this software."
            echo "Type 'interact' at the prompt and try again"
    else

# Setup software paths
#        export ANACONDA_HOME=/home/batmangh/anaconda2
#        export PYTHON_ROOT=/home/batmangh/anaconda2
#        export JUPYTER_RUNTIME_DIR=/home/batmangh/runtime

# create directory. it doesn't hurt anything if it already exists.
#        mkdir -p ~/.jupyter

# Setup Jupyter rundir. Perhaps this step should move to a module.
#        export JUPYTER_RUNTIME_DIR=~/.jupyter

# change the directory, otherwise you are dumped into pwd
#        cd ~/.jupyter

# find the last port that someone used
            declare -i portnum=8888

        portused=$(netstat -nat | grep $portnum)

        while [ ! -z "$portused" ]; do
            echo $portused
            portnum=portnum+1
            portused=$(netstat -nat | grep $portnum)
            echo "new port assigned" $portnum
             done
 

# User instructions
        echo "Your Jupyter Notebook is ready for use."
        echo "------------------------"
        echo "Step 1:"
        echo "Mac/Linux users: launch another terminal and paste the following command:"
        echo "ssh -L $portnum:$HOSTSTR:$portnum bridges.psc.edu -l `whoami`"
        echo "Windows users: run "cmd" then cd to yor PuTTY directory then pased the following command:"
        echo "plink  -L $portnum:$HOSTSTR:$portnum bridges.psc.edu -l `whoami`"
        echo "------------------------"
        echo "Step 2: Open a browser on your computer to http://localhost:$portnum"
        echo "Step 3: Enjoy Jupyter Notebook!"


# Launch the notebook
            jupyter notebook --no-browser --ip=0.0.0.0 --port=$portnum 
    fi
    else
    echo "This a command is to launch a jupyter notebook."
fi
#!/bin/bash`#!/bin/bash

# startupjupyter version 1.1

########### Licensing ####################
#
# Copyright (c) 2017, Pittsburgh Supercomputing Center
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of startupjupyter nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Pittsburgh Supercomputing Center	phone: 412.268.4960  fax: 412.268.5832
# 300 S. Craig Street			e-mail: remarks@psc.edu
# Pittsburgh, PA 15213			web: http://www.psc.edu/
#
######End of Licensing###################

# Software Purpose

# This is a bash script for launching a jupyter notebook from an interactive
# node. It assumes that the user has local ssh terminals on their desktop. 
# This works for GNU/Linux and MacOS users, but Windows does not have terminal
# support by default outside of Cygwin and Ubuntu Windows 10 Feature users.

#store hostname in a string
HOSTSTRPVT=`hostname`
HOSTSTR=$(echo $HOSTSTRPVT | sed 's/pvt/opa/g')
# if the user passes any command args at all, interpret it as a request for help

if [ -z $1 ]; then
# Only do this stuff on an interactive node. 
	if [[ "$HOSTSTR" = br00?.opa.bridges.psc.edu1 ]]; then
		echo "You are on" $HOSTSTR "which is a login node"
		echo "You must be on an interactive node to run this software."
        	echo "Type 'interact' at the prompt and try again"
	else

# Setup software paths
#	    export ANACONDA_HOME=/home/batmangh/anaconda2
#	    export PYTHON_ROOT=/home/batmangh/anaconda2
#	    export JUPYTER_RUNTIME_DIR=/home/batmangh/runtime

# create directory. it doesn't hurt anything if it already exists.
#		mkdir -p ~/.jupyter

# Setup Jupyter rundir. Perhaps this step should move to a module.
#		export JUPYTER_RUNTIME_DIR=~/.jupyter

# change the directory, otherwise you are dumped into pwd
#		cd ~/.jupyter

# find the last port that someone used
        	declare -i portnum=8888

		portused=$(netstat -nat | grep $portnum)

		while [ ! -z "$portused" ]; do
			echo $portused
			portnum=portnum+1
			portused=$(netstat -nat | grep $portnum)
			echo "new port assigned" $portnum
     		done
 

# User instructions
		echo "Your Jupyter Notebook is ready for use."
		echo "------------------------"
		echo "Step 1:"
		echo "Mac/Linux users: launch another terminal and paste the following command:"
		echo "ssh -L $portnum:$HOSTSTR:$portnum bridges.psc.edu -l `whoami`"
		echo "Windows users: run "cmd" then cd to yor PuTTY directory then pased the following command:"
		echo "plink  -L $portnum:$HOSTSTR:$portnum bridges.psc.edu -l `whoami`"
		echo "------------------------"
		echo "Step 2: Open a browser on your computer to http://localhost:$portnum"
		echo "Step 3: Enjoy Jupyter Notebook!"


# Launch the notebook
        	jupyter notebook --no-browser --ip=0.0.0.0 --port=$portnum 
	fi
	else
	echo "This a command is to launch a jupyter notebook."
fi
#!/bin/bash`
