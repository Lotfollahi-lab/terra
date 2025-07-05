import submitit
import argparse
import os
from train import train  # assuming your training function is in train.py
from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser("Submitit for Training Job", add_help=False)
    parser.add_argument('--job_dir', default='./submitit_logs', type=str)
    parser.add_argument('--ngpus', default=4, type=int)
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--timeout', default=1440, type=int)
    parser.add_argument('--partition', default='gpu', type=str)
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        args = self.args

        import yaml
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)

        train_dataset = load_from_disk(args.dataset_path)

        LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

        train(args=config_args,
              train_dataset=train_dataset,
              resume_preempt=args.resume,
              save_folder_path=args.job_dir,
              LOCAL_RANK=LOCAL_RANK)

    def checkpoint(self):
        print("Requeuing job due to timeout.")
        new_trainer = Trainer(self.args)
        return submitit.helpers.DelayedSubmission(new_trainer)


def main():
    args = parse_args()

    executor = submitit.AutoExecutor(folder=args.job_dir)

    executor.update_parameters(
        mem_gb=64,
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ngpus,
        cpus_per_task=8,
        nodes=args.nodes,
        timeout_min=args.timeout,
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        slurm_comment='train_job_submitit',
    )

    trainer = Trainer(args)

    job = executor.submit(trainer)
    print(f"Submitted job_id: {job.job_id}")


if __name__ == '__main__':
    main()
