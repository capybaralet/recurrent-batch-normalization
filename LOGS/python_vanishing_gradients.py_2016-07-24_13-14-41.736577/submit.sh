                #!/bin/sh

                ## Reasonable default values
                # Execute the job from the current working directory.
                #PBS -d /home/capybara/recurrent-batch-normalization

                #All jobs must be submitted with an estimated run time
                #PBS -l walltime=23:59:59

                ## Job name
                #PBS -N dbi_2b5eb492043

                ## log out/err files
                # We cannot use output_file and error_file here for now.
                # We will use dbi_...out-id and dbi_...err-id instead
                # Torque append -{JOBID} to the output filename
                # But not Moab, so we add it
                #PBS -o /home/capybara/recurrent-batch-normalization/LOGS/python_vanishing_gradients.py_2016-07-24_13-14-41.736577/dbi_2b5eb492043.out-${MOAB_JOBARRAYINDEX}
                #PBS -e /home/capybara/recurrent-batch-normalization/LOGS/python_vanishing_gradients.py_2016-07-24_13-14-41.736577/dbi_2b5eb492043.err-${MOAB_JOBARRAYINDEX}


                ## The project name for accounting/permission
                #PBS -A jvb-000-ag

        ## Number of CPU (on the same node) per job
        #PBS -l nodes=1:gpus=1

                ## Execute as many jobs as needed
                #PBS -t 0-0%30

                ## Queue name
                #PBS -q gpu_1
export "THEANO_FLAGS=device=gpu,floatX=float32"
export "JOBDISPATCH_RESUBMIT=msub /home/capybara/recurrent-batch-normalization/LOGS/python_vanishing_gradients.py_2016-07-24_13-14-41.736577/submit.sh"

                ## Variable to put into the environment
                #PBS -v THEANO_FLAGS,JOBDISPATCH_RESUBMIT

                ## Execute the 'launcher' script in bash
                # Bash is needed because we use its "array" data structure
                # the -l flag means it will act like a login shell,
                # and source the .profile, .bashrc, and so on
                /bin/bash -l -e /home/capybara/recurrent-batch-normalization/LOGS/python_vanishing_gradients.py_2016-07-24_13-14-41.736577/launcher
