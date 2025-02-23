#!/bin/bash

echo "Setting up environment for Ubuntu..."

# Step 1: Install Ruby and dependencies
sudo apt update
sudo apt install -y ruby-full build-essential zlib1g-dev gcc make

# Configure gem installation path
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Step 2: Install Jekyll and Bundler
gem install jekyll bundler

# Step 3: Install bundle inside exploring-ai-backend repo
cd /path/to/exploring-ai-backend
bundle install

echo "Environment setup complete for Ubuntu."