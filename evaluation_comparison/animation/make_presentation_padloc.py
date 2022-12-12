import open3d as o3d
import numpy as np
import time
import pickle
import os
import spatialmath

keep_show_lines = True
add_translation = True

vis = o3d.visualization.Visualizer()
for sample in range(1):
    os.makedirs(f'./per_presentation/videos/padloc', exist_ok=True)

    pcd_target = o3d.io.read_point_cloud(f'/mnt/gshared/padloc/res/animations/ransac_matches/001412_target.pcd')
    target_color = np.zeros_like(pcd_target.points)
    target_color[:] = [1, 0, 0]
    pcd_target.colors = o3d.utility.Vector3dVector(target_color)

    orig_source = o3d.io.read_point_cloud(f'/mnt/gshared/padloc/res/animations/ransac_matches/000800_source.pcd')
    original_points = np.asanyarray(orig_source.points).copy()

    # delta_pose = np.load(f'./per_presentation/raw/padloc_gt.npz')['arr_0']
    delta_pose = np.load(f'/mnt/gshared/padloc/res/animations/ransac_matches/tf_pred.npz')['arr_0']
    delta_pose = np.linalg.inv(delta_pose)
    rotmse = spatialmath.SE3(delta_pose, check=False)
    # interpolated_poses = rotmse.interp(spatialmath.SE3(), 100)
    interpolated_poses = spatialmath.SE3().interp(rotmse, 100)
    pcd_source = []
    for i in range(len(interpolated_poses)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(original_points.copy())
        pcd.transform(interpolated_poses[i].A)
        pcd_source.append(pcd)
        source_color = np.zeros_like(pcd_source[-1].points)
        source_color[:] = [0, 1, 0]
        pcd_source[-1].colors = o3d.utility.Vector3dVector(source_color)

    pcd_source_translated = o3d.geometry.PointCloud()
    points_st = np.array(pcd_source[0].points)
    colors_st = np.array(pcd_source[0].colors)
    if add_translation:
        points_st[:, 0] += 75.
    pcd_source_translated.points = o3d.utility.Vector3dVector(points_st)
    pcd_source_translated.colors = o3d.utility.Vector3dVector(colors_st)

    pcd_target_translated = o3d.geometry.PointCloud()
    points_tt = np.array(pcd_target.points)
    colors_tt = np.array(pcd_target.colors)
    if add_translation:
        points_tt[:, 0] -= 75.
    pcd_target_translated.points = o3d.utility.Vector3dVector(points_tt)
    pcd_target_translated.colors = o3d.utility.Vector3dVector(colors_tt)

    with open(f'/mnt/gshared/padloc/res/animations/ransac_matches/lines.pickle', 'rb') as f:
        line_dict = pickle.load(f)
    lines_match = o3d.geometry.LineSet()
    if add_translation:
        line_dict['points'][4096:, 0] += 75.
        line_dict['points'][:4096, 0] -= 75.
    line_dict['lines'] = np.array(line_dict['lines'])
    # line_dict['lines'][:,0] += 4096
    # line_dict['lines'][:,1] -= 4096
    lines_match.points = o3d.utility.Vector3dVector(line_dict['points'])
    lines_match.lines = o3d.utility.Vector2iVector(line_dict['lines'])

    # o3d.visualization.draw_geometries([pcd_target, pcd_source[-1]])

    vis.create_window()
    vis.add_geometry(pcd_target_translated)
    vis.add_geometry(pcd_source_translated)
    vis.add_geometry(lines_match)
    vis.get_render_option().point_size = 1.0
    ctr = vis.get_view_control()
    if add_translation:
        ctr.set_zoom(0.4)
    else:
        ctr.set_zoom(0.6)
    # o3d.visualization.draw_geometries([pcd_source_translated, pcd_target_translated, lines_match])
    # vis.run()
    # vis.destroy_window()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f'./per_presentation/videos/padloc/registration_000.png')
    vis.remove_geometry(pcd_source_translated)
    vis.remove_geometry(pcd_target_translated)
    vis.remove_geometry(lines_match)

    # vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    geometry2 = o3d.geometry.PointCloud()
    geometry.points = pcd_source_translated.points
    geometry.colors = pcd_source_translated.colors
    geometry2.points = pcd_target_translated.points
    geometry2.colors = pcd_target_translated.colors
    vis.add_geometry(geometry)
    vis.add_geometry(geometry2)
    if keep_show_lines:
        vis.add_geometry(lines_match)
    vis.get_render_option().point_size = 1.0
    ctr = vis.get_view_control()
    if add_translation:
        ctr.set_zoom(0.4)
    else:
        ctr.set_zoom(0.6)
    vis.poll_events()
    vis.update_renderer()
    # time.sleep(4)
    vis.capture_screen_image(f'./per_presentation/videos/padloc/registration_001.png')
    if add_translation:
        for i in range(100):
            points_st[:, 0] -= 0.75
            geometry.points = o3d.utility.Vector3dVector(points_st)
            points_tt[:, 0] += 0.75
            geometry2.points = o3d.utility.Vector3dVector(points_tt)

            if keep_show_lines:
                line_dict['points'][4096:, 0] -= 0.75
                line_dict['points'][:4096, 0] += 0.75
                lines_match.points = o3d.utility.Vector3dVector(line_dict['points'])
                vis.update_geometry(lines_match)

            # geometry.colors = pcd_source[i].colors
            vis.update_geometry(geometry)
            vis.update_geometry(geometry2)
            vis.poll_events()
            vis.update_renderer()
            # time.sleep(0.03)
            vis.capture_screen_image(f'./per_presentation/videos/padloc/registration_{i + 2:03d}.png')

    if keep_show_lines:
        orig_lines = np.concatenate([line_dict['points'][4096:].T, np.ones((1, 4096))], 0).copy()

    for i in range(100):
        geometry.points = pcd_source[i].points
        geometry.colors = pcd_source[i].colors

        if keep_show_lines:
            line_dict['points'][4096:] = (interpolated_poses[i].A @ orig_lines).T[:, :3]
            lines_match.points = o3d.utility.Vector3dVector(line_dict['points'])
            vis.update_geometry(lines_match)

        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        # time.sleep(0.03)
        if add_translation:
            vis.capture_screen_image(f'./per_presentation/videos/padloc/registration_{i + 2 + 100:03d}.png')
        else:
            vis.capture_screen_image(f'./per_presentation/videos/padloc/registration_{i + 2:03d}.png')
    vis.remove_geometry(geometry)
    vis.remove_geometry(geometry2)
    # time.sleep(2)