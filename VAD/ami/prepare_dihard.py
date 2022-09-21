#!/usr/bin/env python3

# Copyright (c)  2021  Johns Hopkins University (authors: Desh Raj)
# Apache 2.0
import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomHdf5Writer, combine, SupervisionSegment
from lhotse.recipes import prepare_dihard3


# Torch's multithreaded behavior needs to be disabled or it wastes a lot of CPU and
# slow things down.  Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@contextmanager
def get_executor():
    # We'll either return a process pool or a distributed worker pool.
    # Note that this has to be a context manager because we might use multiple
    # context manager ("with" clauses) inside, and this way everything will
    # free up the resources at the right time.
    try:
        # If this is executed on the CLSP grid, we will try to use the
        # Grid Engine to distribute the tasks.
        # Other clusters can also benefit from that, provided a cluster-specific wrapper.
        # (see https://github.com/pzelasko/plz for reference)
        #
        # The following must be installed:
        # $ pip install dask distributed
        # $ pip install git+https://github.com/pzelasko/plz
        name = subprocess.check_output("hostname -f", shell=True, text=True)
        if name.strip().endswith(".clsp.jhu.edu"):
            import plz
            from distributed import Client

            with plz.setup_cluster(memory="6G") as cluster:
                cluster.scale(80)
                yield Client(cluster)
            return
    except:
        pass
    # No need to return anything - compute_and_store_features
    # will just instantiate the pool itself.
    yield None


def locate_corpus(*corpus_dirs):
    for d in corpus_dirs:
        if os.path.exists(d):
            return d
    print(
        "Please create a place on your system to put the downloaded Librispeech data "
        "and add it to `corpus_dirs`"
    )
    sys.exit(1)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num-jobs", type=int, default=min(15, os.cpu_count()))
    return parser

def supervision_alignment(SupervisionSet,supervision_file,ctm):
    return SupervisionSet.from_json(
        supervision_file
    ).with_alignment_from_ctm(ctm)


def main():
    args = get_parser().parse_args()

    corpus_dir = locate_corpus(
        Path("/export/corpora5/LDC/LDC2020E12/LDC2020E12_Third_DIHARD_Challenge_Development_Data"),
    )
    eval_dir = locate_corpus(
        Path("/export/corpora5/LDC/LDC2021E02_Third_DIHARD_Challenge_Evaluation_Data_Complete")
    )
    annotations_dir = Path("/export/c07/sli218")

    # download_ami(corpus_dir, annotations=annotations_dir, mic="sdm")

    output_dir = Path("exp/data")

    # ctm_dir = Path("/export/c07/sli218/kaldi/egs/ami/s5b/ami_ctm")

    # with open(f'/export/c07/sli218/kaldi/egs/ami/s5b/ami_ctm/dev.ctm', 'r') as f_out:
    #     text=""
    #     for line in f_out:
    #         content = line.strip().split()[:5]
    #         if content[1]=="A":
    #             content[1]="1"
    #         text = text+" ".join(content)+"\n"

    # with open(f'/export/c07/sli218/kaldi/egs/ami/s5b/ami_ctm/dev.ctm', 'w') as f_out:
    #     f_out.write(text)
    # # Replace string
    # content = content.replace("A", "1")
    # # Write new content in write mode 'w'
    # with open(f'/export/c07/sli218/kaldi/egs/ami/s5b/ami_ctm/dev.ctm', 'w') as file:
    #     file.write(content)

    print("Dihard3 manifest preparation:")
    ami_manifests = prepare_dihard3(
        corpus_dir,
        eval_dir,
        # annotations_dir=annotations_dir,
        output_dir=output_dir,
        # mic="sdm",
        # partition="full-corpus",
        # max_pause=0,
    )

    # ami_manifests["supervisions"] = ami_manifests["supervisions"]
    
    print("Feature extraction:")
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    with get_executor() as ex:  # Initialize the executor only once.
        for partition, manifests in ami_manifests.items():
            # ''' print(manifests["supervisions"])
            # manifests["supervisions"]=manifests["supervisions"].from_jsonl("exp/data/supervisions_dev.jsonl").with_alignment_from_ctm(
            #         ctm_dir,
            # )
            if (output_dir / f"cuts_dihard_{partition}.json.gz").is_file():
                print(f"{partition} already exists - skipping.")
                continue
            print("Processing", partition)
            print(manifests.keys())
            cut_set = CutSet.from_manifests(
                recordings=manifests["recordings"],
                supervisions=manifests["supervisions"],
            ).cut_into_windows(duration=5)
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/feats_{partition}",
                # when an executor is specified, make more partitions
                # num_jobs=args.num_jobs if ex is None else min(80, len(cut_set)),
                executor=ex,
                # storage_type=LilcomHdf5Writer,
            )
            # .pad(duration=5.0)
            cut_set.to_json(output_dir / f"cuts_dihard_{partition}.json.gz")


if __name__ == "__main__":
    main()
