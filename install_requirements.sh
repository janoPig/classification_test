#!/bin/bash
sudo apt update && sudo apt upgrade
sudo apt-get install build-essential -y
sudo apt-get install python3-dev -y
sudo apt install python3-pip -y
# llvm 18
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh 18
rm -f llvm.sh
# python and venv
sudo apt install python3.10
sudo apt install python3.10-venv
