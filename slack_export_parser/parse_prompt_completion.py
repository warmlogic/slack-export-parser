import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

PROMPT_END_TOKEN = "\n\n###\n\n"
COMPLETION_END_TOKEN = "<e>"


def parse_channel_data(export_path: Path, channels: list[dict]):
    cols = [
        "channel",
        "thread_ts",
        "ts",
        "user",
        "display_name",
        "text",
        # "reactions",
    ]

    slack_df = pd.DataFrame()

    for channel in channels:
        channel_df = pd.DataFrame()

        day_files = sorted((export_path / channel["name"]).glob("*.json"))
        print(f"Parsing channel {channel['name']}: {len(day_files):,} files")
        for day_file in tqdm(day_files):
            with open(day_file, "r") as fp:
                day_data = json.load(fp)

            _df = pd.json_normalize(day_data, sep="__")
            _df = _df.rename(columns={"user_profile__display_name": "display_name"})
            if "subtype" not in _df:
                _df["subtype"] = np.nan
            _df["text"] = _df["text"].astype(str)
            _df["channel"] = channel["name"]
            _df = _df.loc[(_df["type"] == "message") & _df["subtype"].isna()]
            _df = _df.reindex(columns=cols)
            channel_df = pd.concat([channel_df, _df], axis="index", ignore_index=True)

        # User display name is nan when a file is attached, fill in the display name
        missing_mask = channel_df["display_name"].isna()
        users = (
            channel_df.dropna(subset="display_name")
            .groupby(by=["user"])["display_name"]
            .first()
        )
        channel_df.loc[missing_mask, "display_name"] = channel_df.loc[
            missing_mask, "user"
        ].map(users)

        # Number threads
        thread_id = (
            channel_df.dropna(subset=["thread_ts"]).groupby(by=["thread_ts"]).ngroup()
        )
        channel_df["thread_id"] = thread_id
        # Fill in -1 for messages that aren't part of threads
        channel_df["thread_id"] = channel_df["thread_id"].fillna(-1)

        thread_idx = channel_df["thread_id"] != -1
        print(f"Channel: {channel['name']}")
        print(f"\tUsers: {channel_df['user'].nunique():,}")
        print(f"\tMessages: {channel_df.shape[0]:,}")
        print(f"\tThreads: {channel_df.loc[thread_idx, 'thread_id'].nunique():,}")

        # # Number messages in a thread
        # thread_message_id = (
        #     channel_df.dropna(subset=["thread_ts"]).groupby(by=["thread_ts"]).cumcount()
        # )
        # channel_df["thread_message_id"] = thread_message_id

        # We're going to groupby thread_id to collect prompt and response. A thread's
        # parent message can be the prompt to a completion that wasn't threaded, or a
        # completion to a prior message. Therefore, add a duplicate message without
        # thread_id. We will deduplicate later.
        thread_parents = channel_df.loc[
            channel_df["ts"] == channel_df["thread_ts"]
        ].copy()
        thread_parents["thread_ts"] = np.nan
        channel_df = pd.concat(
            [channel_df, thread_parents], axis="index", ignore_index=True
        )

        slack_df = pd.concat([slack_df, channel_df], axis="index", ignore_index=True)

    # Augment user tag so Slack will interpret it
    slack_df["user"] = slack_df["user"].apply(lambda x: f"<@{x}>")

    slack_df = slack_df.sort_values(by=["ts"]).reset_index(drop=True)

    return slack_df


def get_prompt_completion(
    df: pd.DataFrame,
    display_name: str,
    n_prior: int,
) -> pd.DataFrame:
    """Parse a dataframe into prompt and completion

    A user's own message can be a prompt for their completion
    """
    print("Creating prompt and completion...")

    prompt_completion = pd.DataFrame()
    for _, group in df.groupby("thread_id"):
        user_idx = group["display_name"] == display_name
        user_idx = user_idx.loc[user_idx].index
        user_df = group.loc[user_idx]
        user_df["type"] = "completion"
        user_df["prompt_id"] = user_idx
        # this is always the final message
        user_df["prompt_message_id"] = n_prior + 1

        prev_df = pd.DataFrame()
        for i, n in enumerate(range(1, n_prior + 1)[::-1]):
            prev_msg = group.shift(n).reindex(index=user_idx)
            prev_msg["type"] = "prompt"
            prev_msg["prompt_id"] = user_idx
            prev_msg["prompt_message_id"] = i + 1
            prev_df = pd.concat([prev_df, prev_msg], axis="index", ignore_index=True)

        prompt_completion = pd.concat(
            [prompt_completion, prev_df, user_df], axis="index", ignore_index=True
        )

    prompt_completion = (
        prompt_completion.copy()
        .sort_values(by=["prompt_id", "prompt_message_id", "ts"])
        .reset_index(drop=True)
    )

    # Number prompts sequentially
    prompt_completion["prompt_id"] = prompt_completion.groupby(
        by=["prompt_id"]
    ).ngroup()

    prompt_completion = prompt_completion.dropna(subset=["text"]).copy()

    cols = [
        "channel",
        "ts",
        "user",
        "display_name",
        "text",
        "type",
        "prompt_id",
        "prompt_message_id",
    ]

    prompt_completion = prompt_completion.reindex(columns=cols)

    print("\tDone")

    return prompt_completion


def concat_prompt_completion(
    prompt_and_completion: pd.DataFrame,
    prepend_channel: bool,
    prepend_sender: bool,
    sender_type: str,
) -> pd.DataFrame:
    print("Concatenating prompt and completion...")

    _df = prompt_and_completion.copy()

    prompt_idx = _df["type"] == "prompt"
    completion_idx = _df["type"] == "completion"

    # Include text
    cols = ["text"]
    # Optionally add sender
    if prepend_sender:
        cols = [sender_type] + cols
    # Optionally add channel
    if prepend_channel:
        cols = ["channel"] + cols

    _df["prompt"] = (
        _df.loc[prompt_idx, cols].fillna("").apply(lambda x: " ".join(x), axis=1)
    )

    # Join all prompt messages
    prompts = (
        _df.loc[prompt_idx]
        .groupby(by=["prompt_id"], as_index=False)["prompt"]
        .apply(lambda x: " ".join(x))
    )

    # Select the completion rows
    df_completion = _df.loc[completion_idx].copy()
    df_completion = df_completion.drop(columns=["prompt"])
    df_completion = df_completion.rename(columns={"text": "completion"})

    # Add prompts
    df_completion = df_completion.loc[completion_idx].merge(
        prompts, how="inner", on="prompt_id"
    )

    # Remove any occurrences of the prompt end token
    df_completion["prompt"] = df_completion["prompt"].apply(
        lambda x: x.replace(PROMPT_END_TOKEN.strip(), " ")
    )
    # Append end token to prompt
    df_completion["prompt"] = df_completion["prompt"].apply(
        lambda x: f"{x} {PROMPT_END_TOKEN}"
    )

    # Remove any occurrences of the completion end token
    df_completion["completion"] = df_completion["completion"].apply(
        lambda x: x.replace(COMPLETION_END_TOKEN.strip(), " ")
    )
    # Add a whitespace character to the beginning of the completion and append end token
    # to completion
    df_completion["completion"] = df_completion["completion"].apply(
        lambda x: f" {x} {COMPLETION_END_TOKEN}"
    )

    # Final column selection
    cols = ["channel", "ts", "user", "display_name", "completion", "prompt"]
    df_completion = df_completion.reindex(columns=cols)

    print("\tDone")

    return df_completion


def save_data(
    df: pd.DataFrame, export_path: Path, display_name: str, deduplicate: bool = True
):
    path_out = export_path / "parser_data" / display_name
    if not path_out.exists():
        path_out.mkdir(parents=True, exist_ok=True)

    if deduplicate:
        n_duplicates = df.duplicated(subset=["prompt", "completion"]).sum()
        print(f"Dropping {n_duplicates:,} duplicate rows out of {df.shape[0]:,} total")
        df = df.drop_duplicates(subset=["prompt", "completion"])

    print(f"For user {display_name}, found {df.shape[0]:,} prompts and completions")

    # Save parquet
    parquet_out = path_out / "prompt_completion.parquet"
    print(f"Saving file: {parquet_out}")
    df.to_parquet(parquet_out)
    print("\tDone")

    # Save jsonl
    jsonl_out = path_out / "prompt_completion.jsonl"
    print(f"Saving file: {jsonl_out}")
    df[["prompt", "completion"]].to_json(jsonl_out, orient="records", lines=True)
    print("\tDone")


def main(
    export_dir: str,
    display_name: str,
    n_prior: int = 2,
    prepend_channel: bool = True,
    prepend_sender: bool = True,
    sender_type: str = "user",
):
    assert isinstance(n_prior, int)
    assert prepend_channel in [True, False]
    assert prepend_sender in [True, False]
    assert sender_type in ["display_name", "user"]

    export_path = Path(export_dir)

    with open(export_path / "channels.json", "r") as fp:
        channels = json.load(fp)

    df = parse_channel_data(export_path, channels)

    prompt_and_completion = get_prompt_completion(
        df, display_name=display_name, n_prior=n_prior
    )

    cat_pc = concat_prompt_completion(
        prompt_and_completion,
        prepend_channel=prepend_channel,
        prepend_sender=prepend_sender,
        sender_type=sender_type,
    )

    save_data(cat_pc, export_path, display_name)


if __name__ == "__main__":
    fire.Fire(main)
