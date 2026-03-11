# Passkeys in the Wild: A Systematic Study of FIDO2 User Experience Consistency Across Websites
Passkeys have emerged as a phishing-resistant alternative to passwords, yet little is known about how consistently they are implemented across the web. Prior usability research suggests that cross-site inconsistencies increase cognitive load and hinder adoption, as users carry expectations from one website to another. To address this knowledge gap, we conducted a systematic analysis of passkey user experiences across 111 websites using a structured framework of 28 comparison factors. Our results indicate a moderately consistent passkey user experience, with 80\% of websites falling within a cluster of ''standard implementers''. We also observed substantial variation across sectors and site rankings, as well as underutilization of optional and security-key patterns. These inconsistencies introduce security, privacy, and usability risks despite the formal security and privacy guarantees of passkeys. Our work provides a factor-based comparative framework and an empirical baseline for real-world passkey deployments, and offers design recommendations to inform both future research and the continued evolution of passwordless authentication systems.

## Running the analysis scripts

- Python requirement: Python 3.8+ (use `python3`). Using a virtual environment is recommended but optional.
- Create the `output/` folder before running the scripts (scripts place results there):

```bash
mkdir -p output
```

- From the repository root, run the scripts with:

```bash
python3 code/category-analysis.py
python3 code/cluster-analysis.py
python3 code/euclidean-distance.py
python3 code/rankings-analysis.py
python3 code/shannon-entropy.py
```

- **Dependencies:** This repository includes a `requirements.txt` file at the project root. Install the required packages with:

```bash
python3 -m pip install -r requirements.txt
```

- If a script writes files, check the `output/` directory for results. If a script needs extra setup, see its header comments in the `code/` files.


