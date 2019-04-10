# Get GTK bindings from brew
## Install brew if not already installed
	ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

## Install pygobject3
	brew update
	brew install pygobject3

# Get the rest of the dependencies
## Install miniconda from 
	https://docs.conda.io/en/latest/miniconda.html

## Create conda env
conda env create --name soundimg -f=environment.yml

## Link modules gi and cairo to conda environment
	ln -s /usr/local/lib/python3.6/site-packages/cairo path_to_your_anaconda_dir/envs/soundimg_new/lib/python3.6/site-packages/
	ln -s /usr/local/lib/python3.6/site-packages/gi path_to_your_anaconda_dir/envs/soundimg_new/lib/python3.6/site-packages/


## Start soundimg
	conda activate soundimg
	python start.py

