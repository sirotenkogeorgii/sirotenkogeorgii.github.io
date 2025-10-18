source 'https://rubygems.org'

# Lock Ruby version for local dev consistency (matches GitHub Pages build env)
ruby "3.1.4"

# GitHub Pages ships with a curated set of dependencies for Jekyll
# Using the "github-pages" gem keeps the local environment consistent
# with the production environment on GitHub Pages.
group :jekyll_plugins do
  gem 'github-pages', '~> 232'
end

# Ensure Jekyll `serve` works on Ruby 3.x
gem 'webrick', '~> 1.8'
