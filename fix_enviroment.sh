#!/bin/bash

# Fix package versions in test_env_requirements.txt
source test_env/bin/activate
installed_packages=$(pip freeze)

while IFS= read -r line; do
  if [[ $line != \#* && -n $line ]]; then
    package=$(echo "$line" | cut -d '=' -f 1)
    installed_version=$(echo "$installed_packages" | grep -i "^$package==" | cut -d '=' -f 3)
    if [[ -n $installed_version ]]; then
        echo "$package==$installed_version" >> updated_requirements.txt
    else
        echo missing package: $package
        echo "$line" >> updated_requirements.txt
    fi
  else
    echo "$line" >> updated_requirements.txt
  fi
done < "test_env_requirements.txt"

mv updated_requirements.txt test_env_requirements.txt
