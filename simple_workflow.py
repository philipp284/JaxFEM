import os
import jax
import jax.numpy as np
import pyvista as pv

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, Mesh

# --- 1. Preprocessing: Mesh generation with GMSH ---
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Generate a 3D box mesh
Nx, Ny, Nz = 10, 5, 5
Lx, Ly, Lz = 2.0, 1.0, 1.0
print("Generating mesh with GMSH...")
meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, 
                            domain_x=Lx, domain_y=Ly, domain_z=Lz, 
                            data_dir=data_dir, ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])
print(f"Mesh generated: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements.")

# --- 2. Define Physics Problem ---
# We will use the Poisson equation as a simple example
class Poisson(Problem):
    def get_tensor_map(self):
        return lambda x: x

    def get_mass_map(self):
        def mass_map(u, x):
            # A heat source in the middle of the domain
            val = -np.array([10*np.exp(-(np.power(x[0] - Lx/2, 2) + np.power(x[1] - Ly/2, 2) + np.power(x[2] - Lz/2, 2)) / 0.1)])
            return val
        return mass_map

# --- 3. Define Loads and Boundaries ---
print("Applying boundary conditions...")
def left(point):    return np.isclose(point[0], 0., atol=1e-5)
def right(point):   return np.isclose(point[0], Lx, atol=1e-5)
def bottom(point):  return np.isclose(point[1], 0., atol=1e-5)
def top(point):     return np.isclose(point[1], Ly, atol=1e-5)
def front(point):   return np.isclose(point[2], 0., atol=1e-5)
def back(point):    return np.isclose(point[2], Lz, atol=1e-5)

def dirichlet_val(point):
    return 0.

# Apply zero Dirichlet BC on all faces
location_fns = [left, right, bottom, top, front, back]
value_fns = [dirichlet_val] * 6
vecs = [0] * 6 # 1 degree of freedom for Poisson
dirichlet_bc_info = [location_fns, vecs, value_fns]

# --- 4. Solving ---
print("Solving problem...")
problem = Poisson(mesh=mesh, vec=1, dim=3, ele_type='HEX8', dirichlet_bc_info=dirichlet_bc_info)
sol = solver(problem)

# --- 5. Postprocessing: Visualization ---
vtk_path = os.path.join(data_dir, f'vtk/solution_gmsh.vtu')
os.makedirs(os.path.dirname(vtk_path), exist_ok=True)
save_sol(problem.fes[0], sol[0], vtk_path)
print(f"Solution saved to {vtk_path}")

print("Opening visualizer with PyVista...")
pv_mesh = pv.read(vtk_path)
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, scalars='sol', show_edges=True, cmap='plasma')
plotter.add_text("Jax-FEM Poisson Solution (GMSH Box)", font_size=12)
plotter.show()
