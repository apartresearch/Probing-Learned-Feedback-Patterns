new_paths=("$(pwd)/src/rlhf_model_training" "$(pwd)/src/sparse_codes_training" "$(pwd)/src")

for new_path in "${new_paths[@]}"; do
    # Check if the directory is not already in PYTHONPATH
    if [[ ":$PYTHONPATH:" != *":$new_path:"* ]]; then
        # Add the directory to PYTHONPATH
        export PYTHONPATH="$new_path:$PYTHONPATH"
        echo "Added '$new_path' to PYTHONPATH."
    else
        echo "'$new_path' is already in PYTHONPATH. No changes made."
    fi
done


# Check if the "-v" flag is passed
if [[ "$1" == "-v" ]]; then
    # If "-v" flag is passed, execute pip install with the flag
    venv_directory="sparse_coding_venv"

    # Check if the directory already exists
    if [ ! -d "$venv_directory" ]; then
	    # Create the venv in the specified directory
	    python3 -m venv "$venv_directory"
	    echo "Created a new virtual environment in '$venv_directory'."
    else
	    echo "The directory '$venv_directory' already exists. No virtual environment created."
    fi
    source sparse_coding_venv/bin/activate
    pip install -r requirements.txt
    python3 -m spacy download en_core_web_m
fi
