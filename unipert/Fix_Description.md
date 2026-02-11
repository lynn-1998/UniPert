# **mmseqs Bug Fix:**

- After configuring the `unipert` virtual environment according to the `README.md` in the repository, execute `conda activate unipert` to enter the `unipert` environment, then run `conda install -c bioconda mmseqs2 -y && which mmseqs` (this command installs the officially maintained and more compatible `mmseqs2` command-line tool via bioconda);
- Modify `model.py` located in the `UniPert/unipert` directory by removing the dependency on the problematic `mmseqs` Python package and instead use subprocess to call the system-installed `mmseqs` command-line tool. The modified `model.py` code is shown in `UniPert/unipert/model_fix.py`.

