import argparse
import pandas as pd
import jsonlines
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a codec agent dataset into train, dev, and test sets.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--file_splits_csv", type=str, default=None, 
        help="Path to a CSV file containing file splits. If provided, will use this instead of the ratio arguments."
    )
    parser.add_argument("--train_ratio", type=float, default=0.94, help="Proportion of data to use for training.")
    parser.add_argument("--dev_ratio", type=float, default=0.02, help="Proportion of data to use for development.")
    parser.add_argument("--test_ratio", type=float, default=0.04, help="Proportion of data to use for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling the dataset.")
    args = parser.parse_args()

    metadata_path = args.dataset_path.replace(".txt", "_metadata.jsonl")

    if args.file_splits_csv:
        print(f"Using provided file splits from {args.file_splits_csv}")
        file_splits_df = pd.read_csv(args.file_splits_csv, index_col="file_id")
    else:
        print("No file splits CSV provided, calculating splits based on ratios.")
        # Make sure the ratios are positive and sum to 1
        if args.train_ratio <= 0 or args.dev_ratio <= 0 or args.test_ratio <= 0:
            raise ValueError("Ratios must be positive.")
        if args.train_ratio + args.dev_ratio + args.test_ratio != 1:
            raise ValueError("The sum of train, dev, and test ratios must be 1.")
        # each line is a JSON object with structure:
        # {
        #     "file_id": str,
        #     "interleave_order": str,
        #     "agent_speaker": str,
        #     "example_index": int,
        # }
        metadata_df = pd.read_json(metadata_path, lines=True)

        # first, create a new dataframe with the distinct list of file_ids and the corpus_id, which will be the first part of the file_id
        file_splits_df = metadata_df[["file_id"]].drop_duplicates().reset_index(drop=True)
        file_splits_df["corpus_id"] = file_splits_df["file_id"].apply(lambda x: x.split(os.sep)[0])

        # create a column for the split (train/dev/test)
        file_splits_df["split"] = "train"  # default to train

        # get the unique corpus_ids
        corpus_ids = file_splits_df["corpus_id"].unique().tolist()

        # for each corpus_id, assign the split based on the ratios
        for corpus_id in corpus_ids:
            corpus_df = file_splits_df[file_splits_df["corpus_id"] == corpus_id]
            n = len(corpus_df)

            # shuffle the corpus_df
            corpus_df = corpus_df.sample(frac=1, random_state=args.seed)

            # calculate the split indices
            train_end = max(1, int(n * args.train_ratio))
            dev_end = train_end + max(1, int(n * args.dev_ratio))

            # assign splits
            split_col_ix = corpus_df.columns.get_loc("split")
            corpus_df.iloc[:train_end, split_col_ix] = "train"
            corpus_df.iloc[train_end:dev_end, split_col_ix] = "dev"
            corpus_df.iloc[dev_end:, split_col_ix] = "test"

            # update the original file_splits_df
            file_splits_df.update(corpus_df)

        # Set the index of file_splits_df to be the file_id for efficient lookups
        file_splits_df.set_index("file_id", inplace=True)

        # Save the file_splits_df to a CSV file
        file_splits_csv_path = args.dataset_path.replace(".txt", "_file_splits.csv")
        file_splits_df.to_csv(file_splits_csv_path, index=True)

    # Finally, we can read in the original dataset and split it based on the file_ids. This will be done line-by-line
    # assuming final_metadata_df has the same number of lines as the original dataset and that the rows are in the same order.
    train_path = args.dataset_path.replace(".txt", "_train.txt")
    train_metadata_path = args.dataset_path.replace(".txt", "_train_metadata.jsonl")
    dev_path = args.dataset_path.replace(".txt", "_dev.txt")
    dev_metadata_path = args.dataset_path.replace(".txt", "_dev_metadata.jsonl")
    test_path = args.dataset_path.replace(".txt", "_test.txt")
    test_metadata_path = args.dataset_path.replace(".txt", "_test_metadata.jsonl")
    with (
        open(args.dataset_path, "r", encoding="utf-8") as f,
        open(train_path, "w", encoding="utf-8") as f_train,
        open(dev_path, "w", encoding="utf-8") as f_dev,
        open(test_path, "w", encoding="utf-8") as f_test,
        jsonlines.open(metadata_path, "r") as f_meta,
        jsonlines.open(train_metadata_path, "w") as f_train_meta,
        jsonlines.open(dev_metadata_path, "w") as f_dev_meta,
        jsonlines.open(test_metadata_path, "w") as f_test_meta,
        
    ):
        for line, line_meta in tqdm(zip(f, f_meta)):
            # get the split for the current line
            file_id = line_meta["file_id"]
            split = file_splits_df.loc[file_id, "split"]
            # write the line to the appropriate split file
            if split == "train":
                f_train.write(line)
                f_train_meta.write(line_meta)
            elif split == "dev":
                f_dev.write(line)
                f_dev_meta.write(line_meta)
            elif split == "test":
                f_test.write(line)
                f_test_meta.write(line_meta)

    print("Done!")