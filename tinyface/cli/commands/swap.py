import logging
import os
from functools import partial
from pathlib import Path

from ... import FacePair, TinyFace
from ..filesystem import ensure_directory, is_directory, is_file, list_images
from ..image import read_image, show_image, write_image
from ..tasks import run_tasks


def register_swap_command(subparsers):
    parser = subparsers.add_parser("swap", help="Swap face")
    parser.add_argument(
        "-i", "--input", required=True, help="Path or folder of input image(s)"
    )
    parser.add_argument(
        "-r",
        "--reference",
        required=True,
        help="Path or folder of reference face image(s)",
    )
    parser.add_argument(
        "-d",
        "--destination",
        required=True,
        help="Path or folder of destination face image(s)",
    )
    parser.add_argument("-s", "--save", help="Save path or folder for output image(s)")
    parser.add_argument("-w", "--workers", help="Max job workers")
    parser.set_defaults(func=swap_face)


def check_args(input_path, ref_path, dest_path):
    assert is_file(input_path) or is_directory(input_path), "input path does not exists"
    assert is_file(ref_path) or is_directory(ref_path), "reference path does not exists"
    assert is_file(dest_path) or is_directory(
        dest_path
    ), "destination path does not exists"
    assert (
        (is_file(ref_path) and is_file(dest_path))
        or (is_directory(ref_path) and is_directory(dest_path))
    ), "Both reference and destination paths must either be files or both be directories."


def swap_face(args):
    input_path = args.input
    ref_path = args.reference
    dest_path = args.destination
    save_path = args.save
    max_workers = int(args.workers) if args.workers else None

    check_args(input_path, ref_path, dest_path)

    input_img_paths, face_path_map = get_img_paths(input_path, ref_path, dest_path)

    final_face_names = check_face_names(face_path_map)

    logging.info(
        f"Swapping face in {input_path} using reference {ref_path} and destination {dest_path}"
    )

    tinyface = TinyFace()

    logging.info("Loading models...")
    tinyface.prepare()

    logging.info("Reading faces...")
    face_pairs = get_face_pairs(tinyface, face_path_map, final_face_names)

    final_face_pairs = check_face_pairs(face_pairs)

    img_ref = {"display": None}

    def process_frame(img_path: str):
        _save_path = save_path
        input_img = read_image(img_path)
        res = tinyface.swap_faces(input_img, face_pairs=final_face_pairs)

        if len(input_img_paths) == 1 and not _save_path:
            img_ref["display"] = res
            return

        if is_directory(input_path):
            _save_path = os.path.join(_save_path or "out", Path(img_path).name)

        ensure_directory(_save_path)
        write_image(_save_path, res)

    run_tasks(
        [partial(process_frame, img_path) for img_path in input_img_paths],
        desc="Swapping faces",
        max_workers=max_workers,
    )

    if img_ref.get("display") is not None:
        show_image(img_ref.get("display"))


def get_face_pairs(tinyface, face_path_map, final_face_names):
    return [
        (
            name,
            FacePair(
                reference=tinyface.get_one_face(
                    read_image(face_path_map["references"][name])
                ),
                destination=tinyface.get_one_face(
                    read_image(face_path_map["destinations"][name])
                ),
            ),
        )
        for name in final_face_names
    ]


def check_face_pairs(face_pairs):
    no_ref = []
    no_dest = []
    final_face_pairs = []
    for pair in face_pairs:
        if not pair[1].destination:
            no_dest.append(pair[0])
        elif not pair[1].reference:
            no_ref.append(pair[0])
        else:
            final_face_pairs.append(pair[1])
    if no_ref:
        logging.warning(
            f"({len(no_ref)}) Not found destination face for {', '.join(no_ref)}"
        )
    if no_dest:
        logging.warning(
            f"({len(no_dest)}) Not found reference face for {', '.join(no_dest)}"
        )
    return final_face_pairs


def check_face_names(face_path_map):
    ref_face_names = set(face_path_map["references"].keys())
    dest_face_names = set(face_path_map["destinations"].keys())

    no_ref = []
    no_dest = []
    diff_faces = ref_face_names ^ dest_face_names
    for name in diff_faces:
        if name in ref_face_names:
            no_dest.append(name)
        else:
            no_ref.append(name)
    if no_ref:
        logging.warning(
            f"({len(no_ref)}) Not found destination face for {', '.join(no_ref)}"
        )
    if no_dest:
        logging.warning(
            f"({len(no_dest)}) Not found reference face for {', '.join(no_dest)}"
        )

    final_face_names = ref_face_names & dest_face_names
    assert final_face_names, "No valid face pair found"

    return final_face_names


def get_img_paths(input_path, ref_path, dest_path):
    input_img_paths = (
        list_images(input_path) if is_directory(input_path) else [input_path]
    )
    ref_img_paths = list_images(ref_path) if is_directory(ref_path) else [ref_path]
    dest_img_paths = list_images(dest_path) if is_directory(dest_path) else [dest_path]
    assert input_img_paths, "No input image found"
    assert ref_img_paths, "No reference face found"
    assert dest_img_paths, "No destination face found"

    face_path_map = {"references": {}, "destinations": {}}
    if is_file(ref_path):
        face_path_map["references"]["default"] = ref_path
        face_path_map["destinations"]["default"] = dest_path
    else:
        for path in ref_img_paths:
            face_path_map["references"][Path(path).stem] = path
        for path in dest_img_paths:
            face_path_map["destinations"][Path(path).stem] = path

    return input_img_paths, face_path_map
