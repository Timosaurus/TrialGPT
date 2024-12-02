__author__ = "qiao"

"""
Running TrialGPT matching for multiple patients with lists of trial IDs.
"""

import json
import os
import sys
from TrialGPT import trialgpt_matching


def load_dataset(corpus):
    """Load the dataset containing patient IDs and lists of trial IDs."""
    dataset_path = f"dataset/{corpus}/retrieved_trials.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    with open(dataset_path, "r") as file:
        return json.load(file)  # Load as dictionary


def initialize_output(corpus, model):
    """Initialize the output dictionary or load existing results."""
    output_path = f"results/matching_results_{corpus}_{model}.json"
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            return json.load(file), output_path
    return {}, output_path


def main(corpus, model):
    # Load dataset and initialize output
    dataset = load_dataset(corpus)
    output, output_path = initialize_output(corpus, model)

    # Process each patient in the dataset
    for patient_id, trial_ids in dataset.items():
        # Initialize patient ID in the output
        if patient_id not in output:
            output[patient_id] = {}

        # Process each trial ID
        for trial_id in trial_ids:
            # Skip if already processed
            if trial_id in output[patient_id]:
                continue

            # Perform matching and handle errors gracefully
            try:
                # Example placeholder: Replace with actual trial data retrieval logic if needed
                trial = {"NCTID": trial_id}  # Simplified trial representation
                patient_description = f"Description for patient {patient_id}"  # Placeholder for patient data
                results = trialgpt_matching(trial, patient_description, model)

                # Save results
                output[patient_id][trial_id] = results
                with open(output_path, "w") as file:
                    json.dump(output, file, indent=4)

            except Exception as e:
                print(f"Error processing trial {trial_id} for patient {patient_id}: {e}")

    # Final save
    with open(output_path, "w") as file:
        json.dump(output, file, indent=4)
    print(f"Matching results saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_matching.py <corpus> <model>")
        sys.exit(1)

    corpus_arg = sys.argv[1]
    model_arg = sys.argv[2]

    main(corpus_arg, model_arg)
