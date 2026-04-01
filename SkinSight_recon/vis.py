import time
import numpy as np
import open3d as o3d
import zmq

width=640
height=640

def set_view(vis, extrinsic, intrinsic):
    ctr = vis.get_view_control()
    intrinsic_obj = o3d.camera.PinholeCameraIntrinsic(width, height, width*intrinsic[0][0], height*intrinsic[1][1], width*intrinsic[0][2], height*intrinsic[1][2])
    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = extrinsic 
    param.intrinsic = intrinsic_obj
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

def set_view_ext(vis, extrinsic):
    f = 0.64991707 * 2/3
    ctr = vis.get_view_control()
    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = extrinsic
    param.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, width*f, height*f, width*0.5, height*0.5)
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

def display_zmq_stream(addr="tcp://localhost:1337"):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(addr)
    socket.setsockopt_string(zmq.SUBSCRIBE, "") 
    print(f"Listening on {addr}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Map Visualization", width=width, height=height)
    vis.get_render_option().point_size = 2.0

    p_global = o3d.geometry.PointCloud()
    first_time = True

    while vis.poll_events():
        try:
            message = socket.recv_pyobj(flags=zmq.NOBLOCK)
            #print(message["extrinsics"])
            path = message["path"]
            p = o3d.io.read_point_cloud(path)
            if not p.is_empty():
                print(f"Reading pcd: {path}, Point Num: {len(p.points)}")
                #R = p.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])
                #p.rotate(R, center=(0, 0, 0))
                p_global += p
                if first_time:
                    
                    vis.add_geometry(p_global)
                    vis.reset_view_point(True)
                    first_time = False
                else:
                    vis.update_geometry(p_global)
            else:
                print(f"Warning: {path} not Exist or Empty")
            set_view_ext(vis, np.linalg.inv(message["extrinsics"][-1]))
        except zmq.Again:
            pass
        vis.update_renderer()
        time.sleep(0.01) 

    vis.destroy_window()
    socket.close()
    context.term()
    print("Exit Visualization")

if __name__ == "__main__":
    display_zmq_stream()