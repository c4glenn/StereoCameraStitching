import open3d as o3d
import open3d.visualization as vis

def create_scene() -> list[dict]:
    a_cube = o3d.geometry.TriangleMesh.create_box()
    a_cube.compute_triangle_normals()


    geoms = [{
        "name":"cube",
        "geometry":a_cube
    }]

    return geoms

if __name__ == "__main__":
    geoms = create_scene()
    vis.draw(geoms, bg_color=(0.8, 0.9, 0.9, 1.0), show_ui=True)
