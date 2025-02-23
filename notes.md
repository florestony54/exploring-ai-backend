# Initial setup of environment for serving Jekyll
### Method 1: Script setup
1. navigate to `/setup` in `exploring-ai-backend` 
2. Run setup script
   1. Windows: `win_setup.bat`
   2. Linux: `linux_setup.sh` (**NOTE**: still WIP)

### Method 2: Manual Stepwise Setup
1. install Ruby
    - Ruby Installer (recommended)
      - Go to the [RubyInstaller](https://rubyinstaller.org/) for Windows website.
      - Download the latest version of the Ruby+Devkit installer.
    - via terminal (prob on Ubuntu)
      - `sudo apt install ruby`
      - if on **Ubuntu**: `sudo apt-get install ruby-full build-essential zlib1g-dev` instead
          - THEN: `echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc`
                  `echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc`
                  `echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc`
                  `source ~/.bashrc`
                  ** This will avoid issues related to installing as root
      - install gcc and make
          - `sudo apt install gcc`
          - `sudo apt install make`
2. install jekyll bundler
    - `sudo gem install jekyll bundler`
3. install bundle
    - inside of `exploring-ai-backend` repo
        - run `bundle install` to install jekyll bundles
5

# Serve locally
`bundle exec jekyll serve`
`
# Build
`bundle exec jekyll build`
NOTE: make sure destination in config.yml is set correctly to docs in exploring-ai repo directory

# CSS changes compiled
In another terminal: `cd` into jasper2 then `gulp` to watch for and compile changes
- NOTE: may need to run `sudo apt install gulp`
- ISSUES INSTALLING GULP: https://github.com/orgs/gulpjs/discussions/2488

# Pushing
push changes for both `jasper2` and `docs` to respective repos

# After Pushing
check that custom domain field in github pages says 'exploring-ai.com'. will fail publish if this field gets reset

# Image Ratio
Image ratio 1.6 or 5x3

# Updating
To update to the lastest version installed on your system, run `bundle update --bundler`.
To install the missing version, run `gem install bundler:2.3.7`

# Directory
`cd exploring-ai/jasper2/exploring-ai-backend`

# Quick Start
`cd exploring-ai/jasper2/exploring-ai-backend; bundle exec jekyll serve`

# Repo confusion with submodule
Make sure to check _config.yml that the correct _docs folder is used. Submodule _docs folder wont work


