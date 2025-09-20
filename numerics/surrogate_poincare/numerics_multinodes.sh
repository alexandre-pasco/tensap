# Script directory parameters
scriptpath=. # path where python scripts are located
sbatchscript=${scriptpath}surrogate_poincare.sh
pythonscript=${scriptpath}surrogate_poincare.py
venvpath=./venv/
outputpath=./outputs/ # directory where outputs are saved
mkdir -p ${outputpath} # create the output directory

# submission parameters
partition=SMP-short
time=24:00:00

# Parameters submitted simultaneously
ntrains=("50" "75" "100" "125" "150" "250" "500")


# Additional parameters
benchname=sin_squared_norm 
# choose among exp_mean_sin_exp_cos, borehole, sin_squared_norm, sum_cos_sin_squared_norm
ntest=1000
nseeds=20
m=1
d=8
fitmethod=pymanopt # pymanopt, surrogate, surgreedy
initmethod=as # only for pymanopt. as, random, randlin, surrogate or surgreedy
opti=false # when using surgreedy

# load the python module and source the virtual python env
module purge
module load python
source ${venvpath}bin/activate

for ntrain in ${ntrains[@]}
do
    # directory where outpus will be saved
    prefix=${outputpath}${benchname}_d_${d}_m_${m}/

    if [ "$fitmethod" = "pymanopt" ]
    then
        diroutput=${prefix}fitmethod_${fitmethod}_init_${initmethod}_ntrain_${ntrain}_ntest_${ntest}/
    fi
    
    if [ "$fitmethod" = "surrogate" ]
    then
        diroutput=${prefix}fitmethod_${fitmethod}_ntrain_${ntrain}_ntest_${ntest}/
    fi

    if [ "$fitmethod" = "surgreedy" ]
    then
        diroutput=${prefix}fitmethod_${fitmethod}_opti_${opti}_ntrain_${ntrain}_ntest_${ntest}/
    fi

    # check if folder does not already exists
    if [ ! -e ${diroutput} ]; then 
        
        mkdir -p ${diroutput}

        # submit one job for each key
        # Gather the parameters by type, and concatenate them as name1:val1+nam2:val2...
        bool=opti:${opti}
        int=nseeds:${nseeds}+d:${d}+m:${m}+ntrain:${ntrain}+ntest:${ntest}
        str=benchname:${benchname}+diroutput:${diroutput}+fitmethod:${fitmethod}+initmethod:${initmethod}

        # submit the job
        sbatch --partition ${partition} --time ${time} $sbatchscript $pythonscript $diroutput $int $bool $str >${diroutput}logs_sbatch 2>&1  
        
        echo submitted job, outputs saved at ${diroutput}.

    fi
done
