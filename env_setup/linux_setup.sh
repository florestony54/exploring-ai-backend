#!/bin/bash

LOGFILE=setup_log.txt
echo "Setting up environment for Ubuntu..." > $LOGFILE

# Step 1: Install Ruby and dependencies
echo "Installing Ruby and dependencies..." | tee -a $LOGFILE
sudo apt update | tee -a $LOGFILE
sudo apt install -y ruby-full build-essential zlib1g-dev gcc make | tee -a $LOGFILE

# Configure gem installation path
echo "Configuring gem installation path..." | tee -a $LOGFILE
{
    echo '# Install Ruby Gems to ~/gems'
    echo 'export GEM_HOME="$HOME/gems"'
    echo 'export PATH="$HOME/gems/bin:$PATH"'
} >> ~/.bashrc
source ~/.bashrc

# Step 2: Check if Jekyll is installed
echo "Checking if Jekyll is installed..." | tee -a $LOGFILE
if jekyll -v > /dev/null 2>&1; then
    echo "Jekyll is already installed." | tee -a $LOGFILE
    jekyll -v | tee -a $LOGFILE
else
    echo "Jekyll is not installed or not found. Installing..." | tee -a $LOGFILE
    gem install jekyll | tee -a $LOGFILE
    if [ $? -ne 0 ]; then
        echo "Failed to install Jekyll." | tee -a $LOGFILE
        exit 1
    fi
fi

# Step 3: Check if Bundler is installed
echo "Checking if Bundler is installed..." | tee -a $LOGFILE
if bundler -v > /dev/null 2>&1; then
    echo "Bundler is already installed." | tee -a $LOGFILE
    bundler -v | tee -a $LOGFILE
else
    echo "Installing Bundler..." | tee -a $LOGFILE
    gem install bundler | tee -a $LOGFILE
    if [ $? -ne 0 ]; then
        echo "Failed to install Bundler." | tee -a $LOGFILE
        exit 1
    fi
fi

# Step 4: Install bundle inside exploring-ai-backend repo
echo "Installing bundle inside exploring-ai-backend repo..." | tee -a $LOGFILE
cd ..
if [ $? -ne 0 ]; then
    echo "Failed to change directory to exploring-ai-backend." | tee -a $LOGFILE
    exit 1
fi

bundle install | tee -a $LOGFILE
if [ $? -ne 0 ]; then
    echo "Failed to install bundle." | tee -a $LOGFILE
    exit 1
fi

echo "Environment setup complete for Ubuntu." | tee -a $LOGFILE
echo "Check $LOGFILE for detailed log." | tee -a $LOGFILE