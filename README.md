# faq-virtual-agent

### Pre-requisite

The app requires python3 to be installed in the machine as the programming language used is Python


### Installation Guide

Once the python is installed, install all the dependencies by running the below command.

pip3 install -r requirements.txt

### Steps to be followed

FAQ Virtual Agent can be built by following below steps.

  1. Select a development environment of your choice
  2. Explore and analyse the dataset - Refer the file faqs.csv
  3. Cleanse and get the noise-free data
  4. Compute embeddings for the dataset - sentence or word embeddings
  5. Preprocess the user query
  6. Compute embeddings for user query
  7. Compare and compute similarity score to find a best match
  8. Look up the dataset for displaying answer of a best matched query
  9. Precomputed embeddings can be persisted for later use
  10. Servers and webframeworks can be leveraged for servicing the request as REST API - optional step


### Build your faq virtual agent in 5 minutes

1. Run Jupyter notebook for quick changes and experiments
2. Run virtual_agent.py to start the virtual agent on interactive mode. It also auto-trains the virtual agent for the first time on the dataset faqs.csv
