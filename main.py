import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

import configuration as config
import helpers
import legibility_classifier as lc
from helpers import SimpleTimeRecorder


PY_JERSEY = "/workspace/miniconda3/envs/jersey/bin/python"
PY_CENTROIDS = "/workspace/miniconda3/envs/centroids/bin/python"
PY_VITPOSE = "/workspace/miniconda3/envs/vitpose/bin/python"
PY_PARSEQ = "/workspace/miniconda3/envs/parseq2/bin/python"


def q(value) -> str:
    return shlex.quote(str(value))


def file_ready(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def dir_ready(path: str) -> bool:
    return os.path.isdir(path) and any(True for _ in os.scandir(path))


def list_track_dirs(path: str):
    if not os.path.isdir(path):
        return []
    items = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full) and name.isdigit():
            items.append(name)
    return sorted(items, key=lambda x: int(x))


def list_image_files(path: str):
    exts = {".jpg", ".jpeg", ".png"}
    if not os.path.isdir(path):
        return []
    items = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() in exts:
            items.append(name)
    return sorted(items)


def run_cmd(command: str) -> bool:
    print(f"Run cmd [{command}]")
    result = subprocess.run(command, shell=True)
    return result.returncode == 0


def maybe_skip(path: str, label: str) -> bool:
    if file_ready(path) or dir_ready(path):
        print(f"Skipping {label}: existing output found at {path}")
        return True
    return False


def get_soccer_ball_list(args):
    soccer_ball_list = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["soccer_ball_list"],
    )
    if not file_ready(soccer_ball_list):
        return []
    with open(soccer_ball_list, "r") as f:
        ball_json = json.load(f)
    return ball_json.get("ball_tracks", [])


def get_filtered_tracklets(tracklets, exclude_balls, args):
    if not exclude_balls:
        return tracklets
    ball_list = set(str(x) for x in get_soccer_ball_list(args))
    return [track for track in tracklets if str(track) not in ball_list]


def get_soccer_net_raw_legibility_results(args, use_filtered=True, filter_name="gauss", exclude_balls=True):
    root_dir = config.dataset["SoccerNet"]["root_dir"]
    image_dir = config.dataset["SoccerNet"][args.part]["images"]
    path_to_images = os.path.join(root_dir, image_dir)

    tracklets = list_track_dirs(path_to_images)
    tracklets = get_filtered_tracklets(tracklets, exclude_balls, args)
    results_dict = {x: [] for x in tracklets}

    filtered = None
    if use_filtered:
        if filter_name == "sim":
            path_to_filter_results = os.path.join(
                config.dataset["SoccerNet"]["working_dir"],
                config.dataset["SoccerNet"][args.part]["sim_filtered"],
            )
        else:
            path_to_filter_results = os.path.join(
                config.dataset["SoccerNet"]["working_dir"],
                config.dataset["SoccerNet"][args.part]["gauss_filtered"],
            )
        with open(path_to_filter_results, "r") as f:
            filtered = json.load(f)

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered and filtered is not None:
            images = filtered.get(directory, [])
        else:
            images = list_image_files(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(
            images_full_path,
            config.dataset["SoccerNet"]["legibility_model"],
            threshold=-1,
            arch=config.dataset["SoccerNet"]["legibility_model_arch"],
        )
        results_dict[directory] = track_results

    full_legibile_path = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["raw_legible_result"],
    )
    with open(full_legibile_path, "w") as outfile:
        json.dump(results_dict, outfile)

    return results_dict


def get_soccer_net_legibility_results(args, use_filtered=False, filter_name="sim", exclude_balls=True):
    root_dir = config.dataset["SoccerNet"]["root_dir"]
    image_dir = config.dataset["SoccerNet"][args.part]["images"]
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = list_track_dirs(path_to_images)
    tracklets = get_filtered_tracklets(tracklets, exclude_balls, args)

    filtered = None
    if use_filtered:
        if filter_name == "sim":
            path_to_filter_results = os.path.join(
                config.dataset["SoccerNet"]["working_dir"],
                config.dataset["SoccerNet"][args.part]["sim_filtered"],
            )
        else:
            path_to_filter_results = os.path.join(
                config.dataset["SoccerNet"]["working_dir"],
                config.dataset["SoccerNet"][args.part]["gauss_filtered"],
            )
        with open(path_to_filter_results, "r") as f:
            filtered = json.load(f)

    legible_tracklets = {}
    illegible_tracklets = []

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered and filtered is not None:
            images = filtered.get(directory, [])
        else:
            images = list_image_files(track_dir)

        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(
            images_full_path,
            config.dataset["SoccerNet"]["legibility_model"],
            arch=config.dataset["SoccerNet"]["legibility_model_arch"],
            threshold=0.5,
        )
        legible = list(np.nonzero(track_results))[0]
        if len(legible) == 0:
            illegible_tracklets.append(directory)
        else:
            legible_images = [images_full_path[i] for i in legible]
            legible_tracklets[directory] = legible_images

    full_legibile_path = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["legible_result"],
    )
    with open(full_legibile_path, "w") as outfile:
        json.dump(legible_tracklets, outfile, indent=4)

    full_illegibile_path = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["illegible_result"],
    )
    with open(full_illegibile_path, "w") as outfile:
        json.dump({"illegible": illegible_tracklets}, outfile, indent=4)

    return legible_tracklets, illegible_tracklets


def generate_json_for_pose_estimator(args, legible=None):
    all_files = []

    if legible is not None:
        for key in legible.keys():
            for entry in legible[key]:
                all_files.append(os.path.join(os.getcwd(), entry))
    else:
        root_dir = os.path.join(os.getcwd(), config.dataset["SoccerNet"]["root_dir"])
        image_dir = config.dataset["SoccerNet"][args.part]["images"]
        path_to_images = os.path.join(root_dir, image_dir)
        tracks = list_track_dirs(path_to_images)
        for tr in tracks:
            track_dir = os.path.join(path_to_images, tr)
            imgs = list_image_files(track_dir)
            for img in imgs:
                all_files.append(os.path.join(track_dir, img))

    output_json = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["pose_input_json"],
    )
    helpers.generate_json(all_files, output_json)


def consolidated_results(image_dir, predictions, illegible_path, soccer_ball_list=None):
    merged = dict(predictions)

    if soccer_ball_list is not None and file_ready(soccer_ball_list):
        with open(soccer_ball_list, "r") as sf:
            balls_json = json.load(sf)
        balls_list = balls_json.get("ball_tracks", [])
        for entry in balls_list:
            merged[str(entry)] = 1

    with open(illegible_path, "r") as f:
        illegible_dict = json.load(f)

    all_illegible = illegible_dict.get("illegible", [])
    for entry in all_illegible:
        if str(entry) not in merged:
            merged[str(entry)] = -1

    all_tracks = list_track_dirs(image_dir)
    for t in all_tracks:
        if t not in merged:
            merged[t] = -1
        else:
            merged[t] = int(merged[t])

    return merged


def train_parseq(args):
    if args.dataset == "Hockey":
        print("Train PARSeq for Hockey")
        parseq_dir = config.str_home
        current_dir = os.getcwd()
        os.chdir(parseq_dir)
        data_root = os.path.join(current_dir, config.dataset["Hockey"]["root_dir"], config.dataset["Hockey"]["numbers_data"])
        command = (
            f"{PY_PARSEQ} train.py +experiment=parseq dataset=real "
            f"data.root_dir={q(data_root)} trainer.max_epochs=25 "
            f"pretrained=parseq trainer.devices=1 trainer.val_check_interval=1 "
            f"data.batch_size=128 data.max_label_length=2"
        )
        success = run_cmd(command)
        os.chdir(current_dir)
        print("Done training" if success else "Training failed")
    else:
        print("Train PARSeq for Soccer")
        parseq_dir = config.str_home
        current_dir = os.getcwd()
        os.chdir(parseq_dir)
        data_root = os.path.join(current_dir, config.dataset["SoccerNet"]["root_dir"], config.dataset["SoccerNet"]["numbers_data"])
        command = (
            f"{PY_PARSEQ} train.py +experiment=parseq dataset=real "
            f"data.root_dir={q(data_root)} trainer.max_epochs=25 "
            f"pretrained=parseq trainer.devices=1 trainer.val_check_interval=1 "
            f"data.batch_size=128 data.max_label_length=2"
        )
        success = run_cmd(command)
        os.chdir(current_dir)
        print("Done training" if success else "Training failed")


def hockey_pipeline(args):
    success = True

    if args.pipeline["legible"]:
        root_dir = os.path.join(config.dataset["Hockey"]["root_dir"], config.dataset["Hockey"]["legibility_data"])
        print("Test legibility classifier")
        command = (
            f"{PY_JERSEY} legibility_classifier.py --data {q(root_dir)} "
            f"--arch resnet34 --trained_model {q(config.dataset['Hockey']['legibility_model'])}"
        )
        success = run_cmd(command)
        print("Done legibility classifier")

    if success and args.pipeline["str"]:
        print("Predict numbers")
        current_dir = os.getcwd()
        data_root = os.path.join(current_dir, config.dataset["Hockey"]["root_dir"], config.dataset["Hockey"]["numbers_data"])
        command = f"{PY_PARSEQ} str.py {q(config.dataset['Hockey']['str_model'])} --data_root={q(data_root)}"
        success = run_cmd(command)
        print("Done predict numbers")


def soccer_net_pipeline(args):
    legible_dict = None
    legible_results = None
    consolidated_dict = None
    analysis_results = None

    Path(config.dataset["SoccerNet"]["working_dir"]).mkdir(parents=True, exist_ok=True)
    success = True

    image_dir = os.path.join(
        config.dataset["SoccerNet"]["root_dir"],
        config.dataset["SoccerNet"][args.part]["images"],
    )
    soccer_ball_list = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["soccer_ball_list"],
    )
    features_dir = config.dataset["SoccerNet"][args.part]["feature_output_folder"]
    full_legibile_path = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["legible_result"],
    )
    illegible_path = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["illegible_result"],
    )
    gt_path = os.path.join(
        config.dataset["SoccerNet"]["root_dir"],
        config.dataset["SoccerNet"][args.part]["gt"],
    )
    input_json = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["pose_input_json"],
    )
    output_json = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["pose_output_json"],
    )
    str_result_file = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["jersey_id_result"],
    )
    final_results_path = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["final_result"],
    )
    gauss_filtered_path = os.path.join(
        config.dataset["SoccerNet"]["working_dir"],
        config.dataset["SoccerNet"][args.part]["gauss_filtered"],
    )

    if args.pipeline["soccer_ball_filter"]:
        with SimpleTimeRecorder("Determine soccer ball"):
            print("Determine soccer ball")
            if file_ready(soccer_ball_list) and not args.force:
                print(f"Skipping soccer ball filter: existing output found at {soccer_ball_list}")
            else:
                success = helpers.identify_soccer_balls(image_dir, soccer_ball_list)
            print("Done determine soccer ball")

    if args.pipeline["feat"] and success:
        with SimpleTimeRecorder("Generate features"):
            print("Generate features")
            if dir_ready(features_dir) and not args.force:
                print(f"Skipping feature generation: existing output found at {features_dir}")
            else:
                command = (
                    f"{PY_CENTROIDS} {q(config.reid_script)} "
                    f"--tracklets_folder {q(image_dir)} --output_folder {q(features_dir)}"
                )
                success = run_cmd(command)
            print("Done generating features")

    if args.pipeline["filter"] and success:
        with SimpleTimeRecorder("Identify and remove outliers"):
            print("Identify and remove outliers")
            if file_ready(gauss_filtered_path) and not args.force:
                print(f"Skipping outlier filtering: existing output found at {gauss_filtered_path}")
            else:
                command = f"{PY_JERSEY} gaussian_outliers.py --tracklets_folder {q(image_dir)} --output_folder {q(features_dir)}"
                success = run_cmd(command)
            print("Done removing outliers")

    if args.pipeline["legible"] and success:
        with SimpleTimeRecorder("Classifying Legibility"):
            print("Classifying Legibility:")
            try:
                if file_ready(full_legibile_path) and file_ready(illegible_path) and not args.force:
                    print("Skipping legibility: existing outputs found")
                    with open(full_legibile_path, "r") as f:
                        legible_dict = json.load(f)
                else:
                    legible_dict, _ = get_soccer_net_legibility_results(
                        args,
                        use_filtered=True,
                        filter_name="gauss",
                        exclude_balls=True,
                    )
            except Exception as error:
                print(f"Failed to run legibility classifier: {error}")
                success = False
            print("Done classifying legibility")

    if args.pipeline["legible_eval"] and success:
        with SimpleTimeRecorder("Evaluate Legibility"):
            print("Evaluate Legibility results:")
            try:
                if legible_dict is None:
                    with open(full_legibile_path, "r") as openfile:
                        legible_dict = json.load(openfile)

                helpers.evaluate_legibility(
                    gt_path,
                    illegible_path,
                    legible_dict,
                    soccer_ball_list=soccer_ball_list,
                )
            except Exception as e:
                print(e)
                success = False
            print("Done evaluating legibility")

    if args.pipeline["pose"] and success:
        with SimpleTimeRecorder("Generate json for pose"):
            print("Generating json for pose")
            try:
                if file_ready(input_json) and not args.force:
                    print(f"Skipping pose-input generation: existing output found at {input_json}")
                else:
                    if legible_dict is None:
                        with open(full_legibile_path, "r") as openfile:
                            legible_dict = json.load(openfile)
                    generate_json_for_pose_estimator(args, legible=legible_dict)
            except Exception as e:
                print(e)
                success = False
            print("Done generating json for pose")

        if success:
            with SimpleTimeRecorder("Detect pose"):
                print("Detecting pose")
                if file_ready(output_json) and not args.force:
                    print(f"Skipping pose detection: existing output found at {output_json}")
                else:
                    command = (
                        f"{PY_VITPOSE} pose.py "
                        f"{q(config.pose_home)}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py "
                        f"{q(config.pose_home)}/checkpoints/vitpose-h.pth "
                        f"--img-root / --json-file {q(input_json)} --out-json {q(output_json)}"
                    )
                    success = run_cmd(command)
                print("Done detecting pose")

    if args.pipeline["crops"] and success:
        with SimpleTimeRecorder("Generate crops"):
            print("Generate crops")
            try:
                crops_destination_dir = os.path.join(
                    config.dataset["SoccerNet"]["working_dir"],
                    config.dataset["SoccerNet"][args.part]["crops_folder"],
                    "imgs",
                )
                Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)

                if dir_ready(crops_destination_dir) and not args.force:
                    print(f"Skipping crop generation: existing output found at {crops_destination_dir}")
                else:
                    if legible_results is None:
                        with open(full_legibile_path, "r") as outfile:
                            legible_results = json.load(outfile)
                    helpers.generate_crops(output_json, crops_destination_dir, legible_results)
            except Exception as e:
                print(e)
                success = False
            print("Done generating crops")

    if args.pipeline["str"] and success:
        with SimpleTimeRecorder("Predict numbers"):
            print("Predict numbers")
            image_dir_str = os.path.join(
                config.dataset["SoccerNet"]["working_dir"],
                config.dataset["SoccerNet"][args.part]["crops_folder"],
            )

            if file_ready(str_result_file) and not args.force:
                print(f"Skipping STR: existing output found at {str_result_file}")
            else:
                command = (
                    f"{PY_PARSEQ} str.py {q(config.dataset['SoccerNet']['str_model'])} "
                    f"--data_root={q(image_dir_str)} --batch_size={args.str_batch_size} "
                    f"--inference --result_file {q(str_result_file)}"
                )
                success = run_cmd(command)
            print("Done predict numbers")

    if args.pipeline["combine"] and success:
        with SimpleTimeRecorder("Combine tracklet results"):
            if file_ready(final_results_path) and not args.force:
                print(f"Skipping combine: existing output found at {final_results_path}")
                with open(final_results_path, "r") as f:
                    consolidated_dict = json.load(f)
            else:
                results_dict, analysis_results = helpers.process_jersey_id_predictions(
                    str_result_file,
                    useBias=True,
                )
                consolidated_dict = consolidated_results(
                    image_dir,
                    results_dict,
                    illegible_path,
                    soccer_ball_list=soccer_ball_list,
                )
                with open(final_results_path, "w") as f:
                    json.dump(consolidated_dict, f)

    if args.pipeline["eval"] and success:
        with SimpleTimeRecorder("Evaluate accuracy"):
            if consolidated_dict is None:
                with open(final_results_path, "r") as f:
                    consolidated_dict = json.load(f)
            with open(gt_path, "r") as gf:
                gt_dict = json.load(gf)
            print(len(consolidated_dict.keys()), len(gt_dict.keys()))
            helpers.evaluate_results(consolidated_dict, gt_dict, full_results=analysis_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Options: 'SoccerNet', 'Hockey'")
    parser.add_argument("part", help="Options: 'test', 'val', 'train', 'challenge'")
    parser.add_argument("--train_str", action="store_true", default=False, help="Run training of jersey number recognition")
    parser.add_argument("--force", action="store_true", default=False, help="Force regeneration even if outputs already exist")
    parser.add_argument("--str_batch_size", type=int, default=1, help="Batch size for STR inference")
    args = parser.parse_args()

    if not args.train_str:
        if args.dataset == "SoccerNet":
            actions = {
                "soccer_ball_filter": True,
                "feat": True,
                "filter": True,
                "legible": True,
                "legible_eval": False,
                "pose": True,
                "crops": True,
                "str": True,
                "combine": True,
                "eval": True,
            }
            args.pipeline = actions
            soccer_net_pipeline(args)
        elif args.dataset == "Hockey":
            actions = {
                "legible": True,
                "str": True,
            }
            args.pipeline = actions
            hockey_pipeline(args)
        else:
            print("Unknown dataset")
    else:
        train_parseq(args)
