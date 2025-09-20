# Script directory parameters
scriptpath=./ # path where python scripts are located
sbatchscript=${scriptpath}collective_poincare.sh
pythonscript=${scriptpath}collective_poincare.py
venvpath=./venv/
outputpath=./outputs/ # directory where outputs are saved
mkdir -p ${outputpath} # create the output directory

# submission parameters
partition=SMP-short
time=24:00:00

# Parameters submitted simultaneously
ntrains=("50" "75" "100" "125" "150" "250" "500")
nytrains=("5" "5" "5" "5" "5" "6" "7")

# Additional parameters
benchname=quartic_sin_collective 
# choose among quartic_sin_collective
ntest=1000
nseeds=20
m=3
d=9
nmat=3
indicesy=15
fitmethod=surrogate # pymanopt, surrogate, surgreedy
initmethod=as # only for pymanopt. as, random, randlin, surrogate or surgreedy
opti=false # when using surgreedy

# load the python module and source the virtual python env
module purge
module load python
source ${venvpath}bin/activate

for i in "${!ntrains[@]}";
do 
    ntrain=${ntrains[i]}
    nytrain=${nytrains[i]}
    # directory where outpus will be saved
    prefix=${outputpath}${benchname}_d_${d}_m_${m}_nmat_${nmat}/

    if [ "$fitmethod" = "pymanopt" ]
    then
        diroutput=${prefix}fitmethod_${fitmethod}_init_${initmethod}_ntrain_${ntrain}_nytrain_${nytrain}_ntest_${ntest}/
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
        int=nseeds:${nseeds}+d:${d}+m:${m}+ntrain:${ntrain}+nytrain:${nytrain}+ntest:${ntest}+nmat:${nmat}+indicesy:${indicesy}
        str=benchname:${benchname}+diroutput:${diroutput}+fitmethod:${fitmethod}+initmethod:${initmethod}

        # submit the job
        sbatch --partition ${partition} --time ${time} $sbatchscript $pythonscript $diroutput $int $bool $str >${diroutput}logs_sbatch 2>&1  
        
        echo submitted job, outputs saved at ${diroutput}.

    fi
done
