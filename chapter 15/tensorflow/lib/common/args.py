# -*- coding: utf-8 -*-


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument("--pattern", type=str, help="dataset pattern")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="checkpoint path.")
    parser.add_argument("--model_name", type=str, help="model name.")
    parser.add_argument("--model_dir", type=str, help="saved model dir")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="learning_rate.")
    parser.add_argument("--decay_rate", type=float, default=0.8, help="learning decay rate.")
    parser.add_argument("--decay_steps", type=int, default=100000, help="learning decay steps.")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size.")
    parser.add_argument("--num_epochs", type=int, default=1, help="epoch number.")
    parser.add_argument("--buffer_size", type=int, default=10000, help="buffer size.")
    parser.add_argument("--num_parallel_calls", type=int, help="num parallel calls.")

    parser.add_argument("--save_summary_steps", default=10000, type=int, help="save_summary_steps.")
    parser.add_argument("--save_checkpoints_steps", default=10000, type=int, help="save_checkpoints_steps.")
    parser.add_argument("--keep_checkpoint_max", default=5, type=int, help="keep_checkpoint_max.")
    parser.add_argument("--log_step_count_steps", default=1000, type=int, help="log_step_count_steps.")
    parser.add_argument("--max_steps", default=None, type=int, help="max_steps.")
    parser.add_argument("--eval_steps", default=1000, type=int, help="eval_steps.")
    parser.add_argument("--eval_throttle_secs", default=300, type=int, help="eval_throttle_secs.")

    parser.add_argument("--inter_op_parallelism_threads", type=int,
                        help="number of inter_op_parallelism_threads")
    parser.add_argument("--intra_op_parallelism_threads", type=int,
                        help="number of intra_op_parallelism_threads")
    parser.add_argument("--cpu", type=int, help="number of cpu")
    parser.add_argument("--gpu", type=int, help="number of gpu")

    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers.")
    parser.add_argument("--worker_index", type=int, default=-1, help="worker index.")
