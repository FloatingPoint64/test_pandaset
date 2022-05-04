import os
import click
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

import pandaset
import laspy


def load_liar(dataset_dir, target_cam_name="front_camera"):
    dataset = pandaset.DataSet(dataset_dir)
    seq_list = dataset.sequences()
    for seq_name in tqdm(seq_list, total=len(seq_list)):
        seq_data = dataset[seq_name]

        if seq_data.lidar is None:
            continue

        seq_data.load()

        lidar_obj = seq_data.lidar
        lidar_obj.set_sensor(1)

        cam = seq_data.camera.get(target_cam_name, None)
        if cam is None:
            raise ValueError(f"Error: Wrong camera name: {target_cam_name}")

        semseg = seq_data.semseg

        seq_pc_xyz = np.empty((0, 6), dtype=np.float64)
        seq_pc_rgb = np.empty((0, 3), dtype=np.uint8)
        seq_pc_cls = np.empty((0, 1), dtype=np.uint8)
        for frame_idx in tqdm(range(len(lidar_obj.data)), total=len(lidar_obj.data), desc=seq_name, leave=True):
            pc_xyzitd = lidar_obj.data[frame_idx].to_numpy()
            pt_in_img_2d, pt_in_cam_3d, inner_indices = pandaset.geometry.projection(
                lidar_points=pc_xyzitd[:, :3],
                camera_data=cam[frame_idx],
                camera_pose=cam.poses[frame_idx],
                camera_intrinsics=cam.intrinsics,
                filter_outliers=True,
            )

            pt_2d_id = np.floor(pt_in_img_2d).astype(np.uint)

            img = np.asarray(seq_data.camera[target_cam_name][frame_idx])
            pc_rgb = img[pt_2d_id[:, 1], pt_2d_id[:, 0]]

            if semseg is not None:
                pc_cls_id = semseg[frame_idx].to_numpy()[inner_indices, :]
            else:
                pc_cls_id = np.zeros([len(inner_indices), 1], dtype=np.uint8)

            seq_pc_xyz = np.vstack([seq_pc_xyz, pc_xyzitd[inner_indices]])
            seq_pc_rgb = np.vstack([seq_pc_rgb, pc_rgb])
            seq_pc_cls = np.vstack([seq_pc_cls, pc_cls_id])

            yield seq_name, frame_idx, pc_xyzitd[inner_indices], pc_rgb, pc_cls_id, img

        yield seq_name, "all_frame", seq_pc_xyz, seq_pc_rgb, seq_pc_cls, None
        dataset.unload(seq_name)


def export_as_las(output_path, pc_xyzitd, pc_rgb, pc_cls_id):
    header = laspy.LasHeader(version="1.4", point_format=7)
    las = laspy.LasData(header)

    las.x = pc_xyzitd[:, 0]
    las.y = pc_xyzitd[:, 1]
    las.z = pc_xyzitd[:, 2]

    las.intensity = pc_xyzitd[:, 3].astype(np.uint16)
    las.gps_time = pc_xyzitd[:, 4]

    las.red = pc_rgb[:, 0]
    las.green = pc_rgb[:, 1]
    las.blue = pc_rgb[:, 2]

    las.classification = pc_cls_id.flatten().astype(np.uint8)

    las.write(output_path)


@click.command()
@click.option("--dataset_dir", "-d", default="./pandaset", type=click.Path(resolve_path=True))
@click.option("--output_root_dir", "-o", default="./pandaset_lidar_las")
def run(dataset_dir, output_root_dir):
    dataset_dir = os.path.abspath(dataset_dir)
    output_root_dir = os.path.abspath(output_root_dir)
    os.makedirs(output_root_dir, exist_ok=True)

    lidar_gen = load_liar(dataset_dir)

    video = None
    prev_seq_name = None
    for data in lidar_gen:
        seq_name, frame_idx, pc_xyzitd, pc_rgb, pc_cls_id, img = data

        if prev_seq_name is None or prev_seq_name != seq_name:
            if video is not None:
                video.release()
            output_path = os.path.join(output_root_dir, "movie", f"{seq_name}.mp4")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            video = cv2.VideoWriter(output_path, fourcc, 8.0, (1920, 1080))
            prev_seq_name = seq_name

        if isinstance(frame_idx, int):
            output_path = os.path.join(output_root_dir, seq_name, f"{seq_name}_{frame_idx:06d}.las")
        else:
            output_path = os.path.join(output_root_dir, f"{seq_name}_{frame_idx}.las")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        export_as_las(output_path, pc_xyzitd, pc_rgb, pc_cls_id)

        if img is not None:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if video is not None:
                video.write(img_bgr)

            # img_pil = Image.fromarray(img)
            # output_path = os.path.join(output_dir, "imgs", f"{seq_name}_{frame_idx}.png")
            # os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # img_pil.save(output_path)

    if video is not None:
        video.release()


if __name__ == "__main__":
    run()
