dataset_path = "data/nell-995"
first_fname = "graph"
second_fname = "dev"

if __name__ == "__main__":
    first_file_path = f"{dataset_path}/{first_fname}.txt"
    second_file_path = f"{dataset_path}/{second_fname}.txt"

    with open(first_file_path) as file:
        first_file_content = file.readlines()

    with open(second_file_path) as file:
        second_file_content = file.readlines()

    overlap = list(set(first_file_content) & set(second_file_content))

    print(f"Checking {dataset_path}.")
    print(f"{len(first_file_content)} triplets in {first_fname}.txt")
    print(f"{len(second_file_content)} triplets in {second_fname}.txt")
    print(f"{len(overlap)} overlapping triplets.")
