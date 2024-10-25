# Notice

This project is in early development, the Docker files are not working as of now and a lot is still missing. Not sure how much broke after the cleanup for upload. 

This also only works for the Mistral chat template, I'm only supporting what I use myself.

# MultiModelProxy

An OAI-compatible proxy server that facilitates fast Chain of Thought generation thought prompting by using a smaller model to do the CoT inference before handing it to the (larger) main model.

# Usage

1. Clone the project
2. Create a virtual Python environment
3. Install the required packages from the "requirements.txt"
4. Copy the included "config_sample.yaml" to "config.yaml"
5. Adjust the configuration
6. Run using following command:

`python -m uvicorn src.main:app --host 127.0.0.1 --port 5000`

# License

This project is licensed under AGPLv3.0 (see included LICENSE file). The following clause applies on top of it and overrides any conflicting clauses:

**This project may not be used in a commercial context under any circumstance unless a commercial license has been granted by the owner. This stipulation applies on top of the
AGPLv3 license.**
