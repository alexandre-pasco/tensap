#!/usr/bin/env bash

# Script directory parameters
scriptpath=~/Documents/python/tensap/numerics/surrogate_collective_poincare/ # path where python scripts are located
pythonscript=${scriptpath}numerics_collective.py
venvpath=~/Documents/python/tensap/.venv/
outputpath=${scriptpath}outputs/ # directory where outputs are saved
mkdir -p ${outputpath} # create the output directory

# Parameters submitted simultaneously
ntrains=("50" "75" "100" "125" "150" "250" "500")
nytrains=("5" "5" "5" "5" "5" "5" "5")

# Additional parameters
benchname=quartic_sin_collective 
# choose among quartic_sin_collective
ntest=1000
nseeds=20
m=2
d=9
nmat=3
indicesy="8"
fitmethod=pymanopt # pymanopt, surrogate, surgreedy
initmethod=as # only for pymanopt. as, random, randlin, surrogate or surgreedy
opti=false # when using surgreedy

# load the python module and source the virtual python env
source ${venvpath}bin/activate

for i in "${!ntrains[@]}";
do 
    ntrain=${ntrains[i]}
    nytrain=${nytrains[i]}
    # directory where outpus will be saved
    prefix=${outputpath}${benchname}_d_${d}_m_${m}_nmat_${nmat}/

    if [ "$fitmethod" = "pymanopt" ]
    then
        diroutput=${prefix}fitmethod_${fitmethod}_init_${initmethod}_ntrain_${ntrain}_ntest_${ntest}/
    fi
    
    if [ "$fitmethod" = "surrogate" ]
    then
        diroutput=${prefix}fitmethod_${fitmethod}_ntrain_${ntrain}_nytrain_${nytrain}_ntest_${ntest}/
    fi

    # check if folder does not already exists
    if [ ! -e ${diroutput} ]; then 
        
        mkdir -p ${diroutput}

        # submit one job for each key
        # Gather the parameters by type, and concatenate them as name1:val1+nam2:val2...
        bool=opti:${opti}
        int=nseeds:${nseeds}+d:${d}+m:${m}+ntrain:${ntrain}+nytrain:${nytrain}+ntest:${ntest}+nmat:${nmat}
        str=benchname:${benchname}+diroutput:${diroutput}+fitmethod:${fitmethod}+initmethod:${initmethod}+indicesy:${indicesy}

        # submit the job
        python ${pythonscript} -b $bool -s $str -i $int >${diroutput}logs_python 2>&1
        
        echo submitted job, outputs saved at ${diroutput}.

    fi
done
